#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi

PRIVATE_PATTERN='/test2/|/mnt/public/|172\.[0-9]+\.|25\.0\.1\.35|experiments/|/opt/conda'
if grep -RInE "${PRIVATE_PATTERN}" \
    --exclude='*.pyc' \
    --exclude='validate_release.sh' \
    .; then
  echo "Private or legacy path scan failed." >&2
  exit 1
fi
echo "Private and legacy path scan: clean"

while IFS= read -r -d '' script; do
  bash -n "${script}"
done < <(find . -type f -name '*.sh' -print0)
echo "Shell syntax: ok"

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

files = list(Path(".").rglob("*.py"))
failures = []
for path in files:
    try:
        compile(path.read_bytes(), str(path), "exec")
    except Exception as exc:  # pragma: no cover - release diagnostic
        failures.append((path, exc))

print(f"Python syntax: {len(files) - len(failures)} ok, {len(failures)} failed")
for path, exc in failures:
    print(f"{path}: {exc}")
raise SystemExit(bool(failures))
PY

"${PYTHON_BIN}" -m pytest -q tests/unit
