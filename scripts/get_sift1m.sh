#!/usr/bin/env bash
# Download and unpack the TEXMEX SIFT1M (or siftsmall 10k) dataset into data/.
# Usage:
#   ./scripts/get_sift1m.sh          # downloads SIFT1M (~160.5 MiB)
#   ./scripts/get_sift1m.sh --small  # downloads siftsmall 10k (~5.06 MiB)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
mkdir -p "${DATA_DIR}"

if [[ "${1:-}" == "--small" ]]; then
    ARCHIVE="siftsmall.tar.gz"
    EXPECTED_DIR="siftsmall"
    CHECK_FILE="siftsmall_base.fvecs"
else
    ARCHIVE="sift.tar.gz"
    EXPECTED_DIR="sift"
    CHECK_FILE="sift_base.fvecs"
fi

URL="ftp://ftp.irisa.fr/local/texmex/corpus/${ARCHIVE}"
ARCHIVE_PATH="${DATA_DIR}/${ARCHIVE}"
TARGET_CHECK="${DATA_DIR}/${EXPECTED_DIR}/${CHECK_FILE}"

if [[ -f "${TARGET_CHECK}" ]]; then
    echo "Dataset already present at ${TARGET_CHECK}"
    exit 0
fi

echo "Downloading ${URL} -> ${ARCHIVE_PATH} ..."
if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 -o "${ARCHIVE_PATH}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
    wget -O "${ARCHIVE_PATH}" "${URL}"
else
    echo "Error: neither curl nor wget is installed." >&2
    exit 1
fi

echo "Extracting ${ARCHIVE_PATH} into ${DATA_DIR} ..."
tar -xzf "${ARCHIVE_PATH}" -C "${DATA_DIR}"
rm -f "${ARCHIVE_PATH}"
echo "Done. Unpacked into ${DATA_DIR}/${EXPECTED_DIR}"
