#!/bin/zsh
set -euo pipefail
umask 077

ROOT_DIR="${0:A:h:h}"
DEST_DIR="${PSU_BOST_DEST:-$ROOT_DIR/private_library/external_datasets/psu_bost_flight_body/archives}"
ARCHIVE="molnar-et-al-open-source-bos-tomography-dataset-2025-06.zip"
EXPECTED_BYTES=5117966684
URL="https://www.datacommons.psu.edu/download/engineering/molnar-et-al-open-source-bos-tomography-dataset-of-high-speed-flow-over-a-flight-body-2025/$ARCHIVE"
SEGMENTS="${PSU_BOST_SEGMENTS:-8}"
CHUNK_BYTES="${PSU_BOST_CHUNK_BYTES:-67108864}"

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

if (( SEGMENTS <= 0 || CHUNK_BYTES <= 0 )); then
  echo "PSU_BOST_SEGMENTS and PSU_BOST_CHUNK_BYTES must be positive" >&2
  exit 1
fi

verify_archive() {
  local actual_bytes
  actual_bytes=$(stat -f%z "$ARCHIVE")
  if [[ "$actual_bytes" != "$EXPECTED_BYTES" ]]; then
    echo "Existing archive has wrong size: expected $EXPECTED_BYTES, got $actual_bytes" >&2
    return 1
  fi
  if [[ -f "$ARCHIVE.sha256" ]]; then
    shasum -a 256 -c "$ARCHIVE.sha256"
  else
    shasum -a 256 "$ARCHIVE" > "$ARCHIVE.sha256"
  fi
  unzip -t "$ARCHIVE"
}

extract_if_requested() {
  if [[ "${PSU_BOST_EXTRACT:-0}" == "1" ]]; then
    unzip -n "$ARCHIVE"
  fi
}

if [[ -f "$ARCHIVE" ]]; then
  echo "Existing archive found; validating without downloading again."
  verify_archive
  extract_if_requested
  echo "Existing download verified."
  exit 0
fi

download_segment() {
  local index="$1"
  local start="$2"
  local stop="$3"
  local part="$ARCHIVE.segment-$(printf '%02d' "$index")"
  local next="$part.next"
  local more="$part.more"
  local existing=0
  local next_existing=0
  local more_existing=0
  local request_start cursor wanted got chunk_stop expected_part

  expected_part=$((stop - start + 1))
  if [[ -f "$part" ]]; then
    existing=$(stat -f%z "$part")
  fi
  if (( existing > expected_part )); then
    echo "segment $index is oversized: $existing > $expected_part" >&2
    return 1
  fi

  cursor=$((start + existing))
  while (( cursor <= stop )); do
    chunk_stop=$((cursor + CHUNK_BYTES - 1))
    if (( chunk_stop > stop )); then
      chunk_stop="$stop"
    fi
    wanted=$((chunk_stop - cursor + 1))
    if [[ -f "$next" ]]; then
      next_existing=$(stat -f%z "$next")
    else
      next_existing=0
    fi
    if [[ -f "$more" ]]; then
      more_existing=$(stat -f%z "$more")
      if (( next_existing + more_existing > wanted )); then
        echo "segment $index interrupted chunk is oversized" >&2
        return 1
      fi
      cat "$more" >> "$next"
      rm -f "$more"
      next_existing=$((next_existing + more_existing))
    fi
    if (( next_existing > wanted )); then
      echo "segment $index temporary chunk is oversized" >&2
      return 1
    fi
    request_start=$((cursor + next_existing))
    if (( request_start <= chunk_stop )); then
      rm -f "$more"
      curl \
        --fail \
        --ipv4 \
        --http1.1 \
        --location \
        --retry 12 \
        --retry-all-errors \
        --retry-delay 3 \
        --connect-timeout 30 \
        --range "$request_start-$chunk_stop" \
        --silent \
        --show-error \
        --output "$more" \
        "$URL"
      cat "$more" >> "$next"
      rm -f "$more"
    fi
    got=$(stat -f%z "$next")
    if (( got != wanted )); then
      echo "segment $index chunk size mismatch: expected $wanted, got $got" >&2
      return 1
    fi
    cat "$next" >> "$part"
    rm -f "$next"
    cursor=$((cursor + got))
    echo "segment $index: $((cursor - start)) / $expected_part bytes"
  done

  got=$(stat -f%z "$part")
  if (( got != expected_part )); then
    echo "segment $index final size mismatch: expected $expected_part, got $got" >&2
    return 1
  fi
}

echo "Downloading the 9-view preprocessed NIRT input with $SEGMENTS resumable segments."
echo "Destination: $DEST_DIR/$ARCHIVE"
segment_bytes=$(((EXPECTED_BYTES + SEGMENTS - 1) / SEGMENTS))
typeset -a pids logs
for (( index = 0; index < SEGMENTS; index++ )); do
  start=$((index * segment_bytes))
  stop=$((start + segment_bytes - 1))
  if (( stop >= EXPECTED_BYTES )); then
    stop=$((EXPECTED_BYTES - 1))
  fi
  log="$ARCHIVE.segment-$(printf '%02d' "$index").log"
  download_segment "$index" "$start" "$stop" > "$log" 2>&1 &
  pids+=("$!")
  logs+=("$log")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "At least one segment failed. Recent per-segment logs:" >&2
  for log in "${logs[@]}"; do
    echo "--- $log" >&2
    tail -n 8 "$log" >&2 || true
  done
  exit 1
fi

rm -f "$ARCHIVE.assembling"
for (( index = 0; index < SEGMENTS; index++ )); do
  cat "$ARCHIVE.segment-$(printf '%02d' "$index")" >> "$ARCHIVE.assembling"
done

ACTUAL_BYTES=$(stat -f%z "$ARCHIVE.assembling")
if [[ "$ACTUAL_BYTES" != "$EXPECTED_BYTES" ]]; then
  echo "Size check failed: expected $EXPECTED_BYTES, got $ACTUAL_BYTES" >&2
  exit 1
fi

mv "$ARCHIVE.assembling" "$ARCHIVE"
shasum -a 256 "$ARCHIVE" > "$ARCHIVE.sha256"
verify_archive

for (( index = 0; index < SEGMENTS; index++ )); do
  rm -f "$ARCHIVE.segment-$(printf '%02d' "$index")"
done

extract_if_requested

echo "Download verified. Set PSU_BOST_EXTRACT=1 to extract the 5.23 GB MAT file."
