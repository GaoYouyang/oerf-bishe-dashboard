from __future__ import annotations

import hashlib
import io
import re
import struct
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Mapping

import pytest

from site_tools.fetch_zip_member_range import (
    IntegrityError,
    RangeResponseError,
    UnsafeMemberError,
    ZipMemberSpec,
    download_zip_member,
)


@dataclass
class _FakeResponse:
    payload: bytes
    start: int
    end: int
    total: int
    truncate_to: int | None = None
    fail_after: int | None = None
    status_code: int = 206

    @property
    def headers(self) -> Mapping[str, str]:
        return {
            "Content-Range": f"bytes {self.start}-{self.end}/{self.total}",
            "Content-Length": str(self.end - self.start + 1),
        }

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        payload = self.payload
        if self.truncate_to is not None:
            payload = payload[: self.truncate_to]
        emitted = 0
        for cursor in range(0, len(payload), chunk_size):
            chunk = payload[cursor : cursor + chunk_size]
            if self.fail_after is not None and emitted < self.fail_after < emitted + len(chunk):
                keep = self.fail_after - emitted
                if keep:
                    yield chunk[:keep]
                raise OSError("simulated connection reset")
            yield chunk
            emitted += len(chunk)
            if self.fail_after is not None and emitted >= self.fail_after:
                raise OSError("simulated connection reset")

    def close(self) -> None:
        return None


class _FakeSession:
    def __init__(
        self,
        archive: bytes,
        *,
        data_offset: int,
        truncate_once: int | None = None,
        fail_once: int | None = None,
        always_truncate: int | None = None,
    ) -> None:
        self.archive = archive
        self.data_offset = data_offset
        self.truncate_once = truncate_once
        self.fail_once = fail_once
        self.always_truncate = always_truncate
        self._truncated = False
        self._failed = False
        self.ranges: list[tuple[int, int]] = []

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
        stream: bool,
    ) -> _FakeResponse:
        del url, timeout, stream
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"])
        assert match is not None
        start, end = map(int, match.groups())
        self.ranges.append((start, end))
        assert 0 <= start <= end < len(self.archive)
        truncate_to = None
        fail_after = None
        if start >= self.data_offset:
            if self.always_truncate is not None:
                truncate_to = min(self.always_truncate, end - start)
            elif self.truncate_once is not None and not self._truncated:
                truncate_to = min(self.truncate_once, end - start)
                self._truncated = True
            if self.fail_once is not None and not self._failed:
                fail_after = min(self.fail_once, end - start)
                self._failed = True
        return _FakeResponse(
            payload=self.archive[start : end + 1],
            start=start,
            end=end,
            total=len(self.archive),
            truncate_to=truncate_to,
            fail_after=fail_after,
        )


def _archive_fixture(member_name: str = "nested/member.bin") -> tuple[bytes, bytes, ZipMemberSpec, int]:
    payload = (b"BOST-range-download\x00" * 513) + bytes(range(256)) * 7
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, payload)
        info = archive.getinfo(member_name)
    raw = buffer.getvalue()
    name_length = int.from_bytes(raw[info.header_offset + 26 : info.header_offset + 28], "little")
    extra_length = int.from_bytes(raw[info.header_offset + 28 : info.header_offset + 30], "little")
    data_offset = info.header_offset + 30 + name_length + extra_length
    spec = ZipMemberSpec(
        url="https://example.invalid/archive.zip",
        member_name=member_name,
        local_header_offset=info.header_offset,
        compressed_size=info.compress_size,
        uncompressed_size=info.file_size,
        crc32=info.CRC,
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    return raw, payload, spec, data_offset


def _download(
    spec: ZipMemberSpec,
    destination: Path,
    session: _FakeSession,
    *,
    retries: int = 2,
) -> object:
    return download_zip_member(
        spec,
        destination,
        session=session,
        segment_size=97,
        io_chunk_size=13,
        retries=retries,
        retry_backoff=0,
    )


def test_downloads_fixed_ranges_and_atomically_publishes_member(tmp_path: Path) -> None:
    raw, payload, spec, data_offset = _archive_fixture()
    session = _FakeSession(raw, data_offset=data_offset)
    destination = tmp_path / "output.bin"

    result = _download(spec, destination, session)

    assert destination.read_bytes() == payload
    assert result.status == "downloaded"
    assert result.resumed_from == 0
    data_ranges = [(start, end) for start, end in session.ranges if start >= data_offset]
    expected_ranges = [
        (data_offset + start, data_offset + min(start + 97, spec.compressed_size) - 1)
        for start in range(0, spec.compressed_size, 97)
    ]
    assert data_ranges == expected_ranges
    assert not destination.with_name("output.bin.compressed.part").exists()
    assert not list(tmp_path.glob(".output.bin.*.tmp"))


def test_retries_after_midstream_connection_reset_without_repeating_bytes(tmp_path: Path) -> None:
    raw, payload, spec, data_offset = _archive_fixture()
    session = _FakeSession(raw, data_offset=data_offset, fail_once=19)
    destination = tmp_path / "retry.bin"

    result = _download(spec, destination, session, retries=2)

    assert destination.read_bytes() == payload
    data_starts = [start for start, _ in session.ranges if start >= data_offset]
    assert data_starts[1] == data_starts[0] + 19
    assert result.range_requests > 1


def test_retries_a_silently_truncated_range(tmp_path: Path) -> None:
    raw, payload, spec, data_offset = _archive_fixture()
    session = _FakeSession(raw, data_offset=data_offset, truncate_once=23)
    destination = tmp_path / "truncated.bin"

    _download(spec, destination, session, retries=2)

    assert destination.read_bytes() == payload
    data_starts = [start for start, _ in session.ranges if start >= data_offset]
    assert data_starts[1] == data_starts[0] + 23


def test_resumes_compressed_stream_across_calls(tmp_path: Path) -> None:
    raw, payload, spec, data_offset = _archive_fixture()
    destination = tmp_path / "resume.bin"
    failing = _FakeSession(raw, data_offset=data_offset, always_truncate=31)

    with pytest.raises(RangeResponseError):
        _download(spec, destination, failing, retries=1)

    partial = destination.with_name("resume.bin.compressed.part")
    assert partial.stat().st_size == 31
    assert not destination.exists()

    healthy = _FakeSession(raw, data_offset=data_offset)
    result = _download(spec, destination, healthy, retries=2)

    assert destination.read_bytes() == payload
    data_starts = [start for start, _ in healthy.ranges if start >= data_offset]
    assert data_starts[0] == data_offset + 31
    assert result.resumed_from == 31


def test_crc_failure_never_replaces_existing_destination(tmp_path: Path) -> None:
    raw, _payload, spec, data_offset = _archive_fixture()
    bad_crc = spec.crc32 ^ 0x1
    corrupted_header = bytearray(raw)
    struct.pack_into("<I", corrupted_header, spec.local_header_offset + 14, bad_crc)
    bad_spec = replace(spec, crc32=bad_crc)
    destination = tmp_path / "protected.bin"
    destination.write_bytes(b"previous valid publication")
    session = _FakeSession(bytes(corrupted_header), data_offset=data_offset)

    with pytest.raises(IntegrityError, match="CRC32 mismatch"):
        _download(bad_spec, destination, session)

    assert destination.read_bytes() == b"previous valid publication"
    assert not destination.with_name("protected.bin.compressed.part").exists()


def test_sha256_failure_discards_partial_and_does_not_publish(tmp_path: Path) -> None:
    raw, _payload, spec, data_offset = _archive_fixture()
    bad_spec = replace(spec, sha256="0" * 64)
    destination = tmp_path / "sha.bin"

    with pytest.raises(IntegrityError, match="SHA256 mismatch"):
        _download(bad_spec, destination, _FakeSession(raw, data_offset=data_offset))

    assert not destination.exists()
    assert not destination.with_name("sha.bin.compressed.part").exists()


def test_rejects_zip_slip_member_before_writing(tmp_path: Path) -> None:
    raw, _payload, spec, data_offset = _archive_fixture("../escape.bin")
    claimed_safe_spec = replace(spec, member_name="claimed-safe.bin")
    destination = tmp_path / "safe.bin"

    with pytest.raises(UnsafeMemberError, match="unsafe ZIP member"):
        _download(
            claimed_safe_spec,
            destination,
            _FakeSession(raw, data_offset=data_offset),
        )

    assert not destination.exists()
    assert not (tmp_path.parent / "escape.bin").exists()
