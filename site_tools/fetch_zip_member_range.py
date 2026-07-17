#!/usr/bin/env python3
"""Resume one deflated ZIP member with strict HTTP Range requests.

The caller supplies metadata normally read from the ZIP central directory.  The
downloader verifies that metadata against the member's local file header, keeps
the compressed stream as a resumable sidecar, inflates it with raw DEFLATE, and
publishes the destination only after every integrity check succeeds.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import os
import re
import struct
import time
import urllib.request
import uuid
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator, Mapping, Protocol


LOCAL_FILE_SIGNATURE = 0x04034B50
LOCAL_FILE_HEADER = struct.Struct("<IHHHHHIIIHH")
ZIP64_EXTRA_ID = 0x0001
DEFLATE_METHOD = 8
DEFAULT_SEGMENT_SIZE = 8 * 1024 * 1024
DEFAULT_IO_CHUNK_SIZE = 256 * 1024
STATE_VERSION = 1


class ZipMemberDownloadError(RuntimeError):
    """Base class for deterministic download failures."""


class ZipFormatError(ZipMemberDownloadError):
    """The remote bytes are not a supported ZIP local member."""


class UnsafeMemberError(ZipMemberDownloadError):
    """The member name could escape an extraction root."""


class RangeResponseError(ZipMemberDownloadError):
    """The server did not honor an exact byte-range request."""


class IntegrityError(ZipMemberDownloadError):
    """The compressed or decompressed member failed verification."""


class ResponseLike(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def iter_content(self, chunk_size: int) -> Iterator[bytes]: ...

    def close(self) -> None: ...


class SessionLike(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
        stream: bool,
    ) -> ResponseLike: ...


@dataclass(frozen=True)
class ZipMemberSpec:
    """Central-directory facts required to locate one ZIP member."""

    url: str
    member_name: str
    local_header_offset: int
    compressed_size: int
    uncompressed_size: int
    crc32: int
    sha256: str | None = None

    def normalized(self) -> "ZipMemberSpec":
        sha256 = self.sha256.lower() if self.sha256 else None
        if self.local_header_offset < 0:
            raise ValueError("local_header_offset must be non-negative")
        if self.compressed_size < 0 or self.uncompressed_size < 0:
            raise ValueError("member sizes must be non-negative")
        if not 0 <= self.crc32 <= 0xFFFFFFFF:
            raise ValueError("crc32 must be an unsigned 32-bit integer")
        if sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("sha256 must contain exactly 64 hexadecimal digits")
        _validate_member_name(self.member_name)
        return ZipMemberSpec(
            url=self.url,
            member_name=self.member_name,
            local_header_offset=self.local_header_offset,
            compressed_size=self.compressed_size,
            uncompressed_size=self.uncompressed_size,
            crc32=self.crc32,
            sha256=sha256,
        )


@dataclass(frozen=True)
class LocalMemberHeader:
    member_name: str
    flags: int
    compression_method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    data_offset: int


@dataclass(frozen=True)
class DownloadResult:
    destination: str
    member_name: str
    compressed_size: int
    uncompressed_size: int
    crc32: str
    sha256: str
    resumed_from: int
    range_requests: int
    status: str


class _UrllibResponse:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.status_code = int(response.status)
        self.headers = response.headers

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        while True:
            chunk = self._response.read(chunk_size)
            if not chunk:
                return
            yield chunk

    def close(self) -> None:
        self._response.close()


class UrllibSession:
    """Small requests-compatible adapter backed only by the standard library."""

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
        stream: bool,
    ) -> _UrllibResponse:
        del stream
        request = urllib.request.Request(url, headers=dict(headers))
        return _UrllibResponse(urllib.request.urlopen(request, timeout=timeout))


def _validate_member_name(member_name: str) -> None:
    if not member_name or "\x00" in member_name or "\\" in member_name:
        raise UnsafeMemberError(f"unsafe ZIP member name: {member_name!r}")
    path = PurePosixPath(member_name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafeMemberError(f"unsafe ZIP member name: {member_name!r}")
    if re.match(r"^[A-Za-z]:", member_name):
        raise UnsafeMemberError(f"unsafe ZIP member name: {member_name!r}")


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value
    return None


def _validate_range_response(response: ResponseLike, start: int, end: int) -> None:
    if response.status_code != 206:
        raise RangeResponseError(
            f"server returned HTTP {response.status_code}, expected 206 for bytes={start}-{end}"
        )
    content_range = _header_value(response.headers, "Content-Range")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range or "")
    if not match or int(match.group(1)) != start or int(match.group(2)) != end:
        raise RangeResponseError(
            f"invalid Content-Range {content_range!r}, expected bytes {start}-{end}/*"
        )
    content_length = _header_value(response.headers, "Content-Length")
    expected = end - start + 1
    if content_length is not None and int(content_length) != expected:
        raise RangeResponseError(
            f"invalid Content-Length {content_length!r}, expected {expected}"
        )


def _open_range(
    session: SessionLike,
    url: str,
    start: int,
    end: int,
    timeout: float,
) -> ResponseLike:
    response = session.get(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
            "User-Agent": "oerf-zip-member-range/1.0",
        },
        timeout=timeout,
        stream=True,
    )
    try:
        _validate_range_response(response, start, end)
    except Exception:
        response.close()
        raise
    return response


def _read_exact_range(
    session: SessionLike,
    url: str,
    start: int,
    end: int,
    *,
    timeout: float,
    retries: int,
    retry_backoff: float,
    io_chunk_size: int,
) -> bytes:
    expected = end - start + 1
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        response: ResponseLike | None = None
        try:
            response = _open_range(session, url, start, end, timeout)
            chunks: list[bytes] = []
            received = 0
            for chunk in response.iter_content(io_chunk_size):
                if not chunk:
                    continue
                received += len(chunk)
                if received > expected:
                    raise RangeResponseError("range response exceeded its declared length")
                chunks.append(chunk)
            if received != expected:
                raise RangeResponseError(
                    f"truncated range bytes={start}-{end}: received {received} of {expected}"
                )
            return b"".join(chunks)
        except Exception as exc:
            last_error = exc
            if attempt < retries and retry_backoff:
                time.sleep(retry_backoff * attempt)
        finally:
            if response is not None:
                response.close()
    assert last_error is not None
    raise RangeResponseError(f"range bytes={start}-{end} failed after {retries} attempts") from last_error


def _zip64_sizes(extra: bytes, need_uncompressed: bool, need_compressed: bool) -> tuple[int | None, int | None]:
    cursor = 0
    while cursor + 4 <= len(extra):
        field_id, field_size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        payload = extra[cursor : cursor + field_size]
        if len(payload) != field_size:
            raise ZipFormatError("truncated ZIP extra field")
        cursor += field_size
        if field_id != ZIP64_EXTRA_ID:
            continue
        position = 0
        uncompressed = compressed = None
        if need_uncompressed:
            if position + 8 > len(payload):
                raise ZipFormatError("ZIP64 extra field lacks uncompressed size")
            uncompressed = struct.unpack_from("<Q", payload, position)[0]
            position += 8
        if need_compressed:
            if position + 8 > len(payload):
                raise ZipFormatError("ZIP64 extra field lacks compressed size")
            compressed = struct.unpack_from("<Q", payload, position)[0]
        return uncompressed, compressed
    if need_uncompressed or need_compressed:
        raise ZipFormatError("ZIP64 sentinel present without ZIP64 size extra field")
    return None, None


def read_local_member_header(
    spec: ZipMemberSpec,
    *,
    session: SessionLike | None = None,
    timeout: float = 90.0,
    retries: int = 3,
    retry_backoff: float = 0.5,
    io_chunk_size: int = DEFAULT_IO_CHUNK_SIZE,
) -> LocalMemberHeader:
    """Fetch, parse, and validate the member's ZIP local file header."""

    spec = spec.normalized()
    if retries < 1:
        raise ValueError("retries must be at least one")
    session = session or UrllibSession()
    fixed = _read_exact_range(
        session,
        spec.url,
        spec.local_header_offset,
        spec.local_header_offset + LOCAL_FILE_HEADER.size - 1,
        timeout=timeout,
        retries=retries,
        retry_backoff=retry_backoff,
        io_chunk_size=io_chunk_size,
    )
    (
        signature,
        _version,
        flags,
        method,
        _mtime,
        _mdate,
        local_crc,
        local_compressed,
        local_uncompressed,
        name_length,
        extra_length,
    ) = LOCAL_FILE_HEADER.unpack(fixed)
    if signature != LOCAL_FILE_SIGNATURE:
        raise ZipFormatError(f"invalid local file signature 0x{signature:08x}")
    if flags & 0x1:
        raise ZipFormatError("encrypted ZIP members are not supported")
    if method != DEFLATE_METHOD:
        raise ZipFormatError(f"compression method {method} is not raw DEFLATE")
    variable_length = name_length + extra_length
    variable = b""
    if variable_length:
        variable_start = spec.local_header_offset + LOCAL_FILE_HEADER.size
        variable = _read_exact_range(
            session,
            spec.url,
            variable_start,
            variable_start + variable_length - 1,
            timeout=timeout,
            retries=retries,
            retry_backoff=retry_backoff,
            io_chunk_size=io_chunk_size,
        )
    name_bytes = variable[:name_length]
    extra = variable[name_length:]
    encoding = "utf-8" if flags & 0x800 else "cp437"
    try:
        member_name = name_bytes.decode(encoding)
    except UnicodeDecodeError as exc:
        raise ZipFormatError(f"member name is not valid {encoding}") from exc
    _validate_member_name(member_name)
    if member_name != spec.member_name:
        raise ZipFormatError(
            f"local header member {member_name!r} does not match {spec.member_name!r}"
        )

    zip64_uncompressed, zip64_compressed = _zip64_sizes(
        extra,
        local_uncompressed == 0xFFFFFFFF,
        local_compressed == 0xFFFFFFFF,
    )
    if zip64_uncompressed is not None:
        local_uncompressed = zip64_uncompressed
    if zip64_compressed is not None:
        local_compressed = zip64_compressed

    uses_descriptor = bool(flags & 0x8)
    if not uses_descriptor or local_compressed not in {0, 0xFFFFFFFF}:
        if local_compressed != spec.compressed_size:
            raise ZipFormatError("local-header compressed size disagrees with central metadata")
    if not uses_descriptor or local_uncompressed not in {0, 0xFFFFFFFF}:
        if local_uncompressed != spec.uncompressed_size:
            raise ZipFormatError("local-header uncompressed size disagrees with central metadata")
    if not uses_descriptor or local_crc != 0:
        if local_crc != spec.crc32:
            raise ZipFormatError("local-header CRC32 disagrees with central metadata")

    return LocalMemberHeader(
        member_name=member_name,
        flags=flags,
        compression_method=method,
        crc32=spec.crc32,
        compressed_size=spec.compressed_size,
        uncompressed_size=spec.uncompressed_size,
        data_offset=spec.local_header_offset + LOCAL_FILE_HEADER.size + variable_length,
    )


def _state_payload(spec: ZipMemberSpec, data_offset: int) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "url": spec.url,
        "member_name": spec.member_name,
        "local_header_offset": spec.local_header_offset,
        "data_offset": data_offset,
        "compressed_size": spec.compressed_size,
        "uncompressed_size": spec.uncompressed_size,
        "crc32": f"{spec.crc32:08x}",
        "sha256": spec.sha256,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _prepare_partial(partial: Path, state_path: Path, expected_state: Mapping[str, Any]) -> int:
    reset = False
    if partial.exists() != state_path.exists():
        reset = True
    elif partial.exists():
        try:
            observed_state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            reset = True
        else:
            reset = observed_state != expected_state
    if reset:
        partial.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
    if not partial.exists():
        partial.touch()
        _write_json_atomic(state_path, expected_state)
    size = partial.stat().st_size
    if size > int(expected_state["compressed_size"]):
        partial.unlink()
        state_path.unlink(missing_ok=True)
        partial.touch()
        _write_json_atomic(state_path, expected_state)
        size = 0
    return size


def _append_exact_range(
    session: SessionLike,
    url: str,
    start: int,
    end: int,
    partial: Path,
    *,
    timeout: float,
    io_chunk_size: int,
) -> None:
    expected = end - start + 1
    response = _open_range(session, url, start, end, timeout)
    received = 0
    try:
        with partial.open("ab", buffering=0) as handle:
            for chunk in response.iter_content(io_chunk_size):
                if not chunk:
                    continue
                if received + len(chunk) > expected:
                    raise RangeResponseError("range response exceeded its declared length")
                handle.write(chunk)
                received += len(chunk)
            os.fsync(handle.fileno())
        if received != expected:
            raise RangeResponseError(
                f"truncated range bytes={start}-{end}: received {received} of {expected}"
            )
    finally:
        response.close()


def _download_compressed_stream(
    spec: ZipMemberSpec,
    header: LocalMemberHeader,
    partial: Path,
    *,
    session: SessionLike,
    segment_size: int,
    io_chunk_size: int,
    timeout: float,
    retries: int,
    retry_backoff: float,
) -> tuple[int, int]:
    resumed_from = partial.stat().st_size
    requests = 0
    while partial.stat().st_size < spec.compressed_size:
        relative_start = partial.stat().st_size
        segment_boundary = min(
            ((relative_start // segment_size) + 1) * segment_size,
            spec.compressed_size,
        )
        failures = 0
        while partial.stat().st_size < segment_boundary:
            relative_start = partial.stat().st_size
            start = header.data_offset + relative_start
            end = header.data_offset + segment_boundary - 1
            try:
                requests += 1
                _append_exact_range(
                    session,
                    spec.url,
                    start,
                    end,
                    partial,
                    timeout=timeout,
                    io_chunk_size=io_chunk_size,
                )
                failures = 0
            except Exception as exc:
                failures += 1
                if failures >= retries:
                    raise RangeResponseError(
                        f"compressed range bytes={start}-{end} failed after {failures} attempts"
                    ) from exc
                if retry_backoff:
                    time.sleep(retry_backoff * failures)
    final_size = partial.stat().st_size
    if final_size != spec.compressed_size:
        raise IntegrityError(
            f"compressed size mismatch: {final_size} != {spec.compressed_size}"
        )
    return resumed_from, requests


def _inflate_and_verify(
    compressed: Path,
    output: BinaryIO,
    spec: ZipMemberSpec,
    *,
    io_chunk_size: int,
) -> tuple[int, int, str]:
    inflater = zlib.decompressobj(-zlib.MAX_WBITS)
    crc = 0
    digest = hashlib.sha256()
    uncompressed_size = 0
    compressed_size = 0
    with compressed.open("rb") as source:
        while True:
            chunk = source.read(io_chunk_size)
            if not chunk:
                break
            compressed_size += len(chunk)
            try:
                payload = inflater.decompress(chunk)
            except zlib.error as exc:
                raise IntegrityError("raw-DEFLATE stream is invalid") from exc
            if inflater.unused_data:
                raise IntegrityError("compressed member contains trailing bytes")
            if payload:
                output.write(payload)
                uncompressed_size += len(payload)
                crc = binascii.crc32(payload, crc)
                digest.update(payload)
        try:
            payload = inflater.flush()
        except zlib.error as exc:
            raise IntegrityError("raw-DEFLATE stream could not be finalized") from exc
        if payload:
            output.write(payload)
            uncompressed_size += len(payload)
            crc = binascii.crc32(payload, crc)
            digest.update(payload)

    crc &= 0xFFFFFFFF
    sha256 = digest.hexdigest()
    if compressed_size != spec.compressed_size:
        raise IntegrityError(
            f"compressed size mismatch: {compressed_size} != {spec.compressed_size}"
        )
    if not inflater.eof or inflater.unconsumed_tail:
        raise IntegrityError("raw-DEFLATE stream ended before its final block")
    if uncompressed_size != spec.uncompressed_size:
        raise IntegrityError(
            f"uncompressed size mismatch: {uncompressed_size} != {spec.uncompressed_size}"
        )
    if crc != spec.crc32:
        raise IntegrityError(f"CRC32 mismatch: {crc:08x} != {spec.crc32:08x}")
    if spec.sha256 is not None and sha256 != spec.sha256:
        raise IntegrityError(f"SHA256 mismatch: {sha256} != {spec.sha256}")
    return uncompressed_size, crc, sha256


def _verify_existing(path: Path, spec: ZipMemberSpec, io_chunk_size: int) -> tuple[int, str] | None:
    if not path.is_file() or path.stat().st_size != spec.uncompressed_size:
        return None
    crc = 0
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(io_chunk_size), b""):
            crc = binascii.crc32(chunk, crc)
            digest.update(chunk)
    crc &= 0xFFFFFFFF
    sha256 = digest.hexdigest()
    if crc != spec.crc32 or (spec.sha256 is not None and sha256 != spec.sha256):
        return None
    return crc, sha256


def download_zip_member(
    spec: ZipMemberSpec,
    destination: Path,
    *,
    session: SessionLike | None = None,
    segment_size: int = DEFAULT_SEGMENT_SIZE,
    io_chunk_size: int = DEFAULT_IO_CHUNK_SIZE,
    timeout: float = 90.0,
    retries: int = 3,
    retry_backoff: float = 0.5,
) -> DownloadResult:
    """Download one ZIP member and atomically replace ``destination``.

    ``sha256``, when present, is the digest of the uncompressed member.  A
    failed network transfer retains ``*.compressed.part`` for the next call;
    an integrity failure discards it so corrupt bytes are never trusted again.
    """

    spec = spec.normalized()
    destination = Path(destination)
    if segment_size < 1 or io_chunk_size < 1:
        raise ValueError("segment_size and io_chunk_size must be positive")
    if retries < 1:
        raise ValueError("retries must be at least one")
    destination.parent.mkdir(parents=True, exist_ok=True)
    session = session or UrllibSession()
    header = read_local_member_header(
        spec,
        session=session,
        timeout=timeout,
        retries=retries,
        retry_backoff=retry_backoff,
        io_chunk_size=io_chunk_size,
    )

    existing = _verify_existing(destination, spec, io_chunk_size)
    if existing is not None:
        crc, sha256 = existing
        return DownloadResult(
            destination=str(destination),
            member_name=spec.member_name,
            compressed_size=spec.compressed_size,
            uncompressed_size=spec.uncompressed_size,
            crc32=f"{crc:08x}",
            sha256=sha256,
            resumed_from=spec.compressed_size,
            range_requests=0,
            status="present",
        )

    partial = destination.with_name(f"{destination.name}.compressed.part")
    state_path = destination.with_name(f"{destination.name}.compressed.part.json")
    expected_state = _state_payload(spec, header.data_offset)
    _prepare_partial(partial, state_path, expected_state)
    resumed_from = partial.stat().st_size

    range_requests = 0
    try:
        resumed_from, range_requests = _download_compressed_stream(
            spec,
            header,
            partial,
            session=session,
            segment_size=segment_size,
            io_chunk_size=io_chunk_size,
            timeout=timeout,
            retries=retries,
            retry_backoff=retry_backoff,
        )
    except RangeResponseError:
        raise

    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as output:
            uncompressed_size, crc, sha256 = _inflate_and_verify(
                partial,
                output,
                spec,
                io_chunk_size=io_chunk_size,
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        raise
    partial.unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)
    return DownloadResult(
        destination=str(destination),
        member_name=spec.member_name,
        compressed_size=spec.compressed_size,
        uncompressed_size=uncompressed_size,
        crc32=f"{crc:08x}",
        sha256=sha256,
        resumed_from=resumed_from,
        range_requests=range_requests,
        status="downloaded",
    )


def _parse_crc32(value: str) -> int:
    cleaned = value.lower().removeprefix("0x")
    if not re.fullmatch(r"[0-9a-f]{1,8}", cleaned):
        raise argparse.ArgumentTypeError("CRC32 must be one to eight hexadecimal digits")
    return int(cleaned, 16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("member_name")
    parser.add_argument("local_header_offset", type=int)
    parser.add_argument("compressed_size", type=int)
    parser.add_argument("uncompressed_size", type=int)
    parser.add_argument("crc32", type=_parse_crc32)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--segment-size", type=int, default=DEFAULT_SEGMENT_SIZE)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    result = download_zip_member(
        ZipMemberSpec(
            url=args.url,
            member_name=args.member_name,
            local_header_offset=args.local_header_offset,
            compressed_size=args.compressed_size,
            uncompressed_size=args.uncompressed_size,
            crc32=args.crc32,
            sha256=args.sha256,
        ),
        args.destination,
        segment_size=args.segment_size,
        timeout=args.timeout,
        retries=args.retries,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
