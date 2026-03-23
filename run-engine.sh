#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
STAMP_FILE="${VENV_DIR}/.phase1-installed"
PORT="7865"
NOAUTOOPEN=0
NOCHECK=0
UPDATE=0

pick_python() {
  local candidates=(
    "${ROOT_DIR}/.venv/bin/python3"
    "${ROOT_DIR}/.venv/bin/python"
    "python3.11"
    "python3.10"
    "/usr/bin/python3"
    "python3.9"
    "python3.12"
    "python3"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if ! command -v "${candidate}" >/dev/null 2>&1; then
      continue
    fi

    local resolved
    resolved="$(command -v "${candidate}")"
    local version_output
    version_output="$("${resolved}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)" || continue

    case "${version_output}" in
      3.9|3.10|3.11)
        echo "${resolved}"
        return 0
        ;;
    esac
  done

  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="${2:-7865}"
      shift 2
      ;;
    --noautoopen)
      NOAUTOOPEN=1
      shift
      ;;
    --nocheck)
      NOCHECK=1
      shift
      ;;
    --update)
      UPDATE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if ! PYTHON_BIN="$(pick_python)"; then
  echo "No compatible Python runtime found. Install Python 3.9, 3.10, or 3.11 before launching the engine." >&2
  exit 1
fi

PYTHON_VERSION="$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  VENV_VERSION="$("${VENV_DIR}/bin/python" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
else
  VENV_VERSION=""
fi

if [[ -d "${VENV_DIR}" && -n "${VENV_VERSION}" && "${VENV_VERSION}" != "${PYTHON_VERSION}" ]]; then
  echo "Existing virtualenv uses Python ${VENV_VERSION}; recreating with ${PYTHON_VERSION}." >&2
  mv "${VENV_DIR}" "${VENV_DIR}.py${VENV_VERSION//./}-bak-$(date +%s)"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment in ${VENV_DIR}" >&2
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade "pip<24.1" "setuptools<71" wheel >/dev/null

if [[ ! -f "${STAMP_FILE}" || "${ROOT_DIR}/requirements/phase1-macos.txt" -nt "${STAMP_FILE}" ]]; then
  echo "Installing engine dependencies..." >&2
  python -m pip install -r "${ROOT_DIR}/requirements/phase1-macos.txt"
  touch "${STAMP_FILE}"
fi

mkdir -p "${ROOT_DIR}/TEMP" "${ROOT_DIR}/logs" "${ROOT_DIR}/opt"

ARGS=(phase1_api.py --pycmd python --port "${PORT}")
if [[ "${NOCHECK}" == "1" ]]; then
  ARGS+=(--nocheck)
fi
if [[ "${UPDATE}" == "1" ]]; then
  ARGS+=(--update)
fi

export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

cd "${ROOT_DIR}"
exec python "${ARGS[@]}"
