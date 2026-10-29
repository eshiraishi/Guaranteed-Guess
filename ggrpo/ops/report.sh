#!/usr/bin/env bash
# Regenerate the report's numbers and figures, build report/midpoint/main.pdf, and list what is still open.
# Run from ggrpo/: bash ops/report.sh. When status.tex sets \booltrue{submit}, a missing number, a \todo left in
# main.tex or an undefined reference fails the build.
set -euo pipefail
uv run python -m ggrpo.latex
cd report/midpoint
if ! tectonic -X compile main.tex --keep-logs > tectonic.out 2>&1; then
  grep -E "^error" tectonic.out; grep -n -A3 "^!" main.log
  echo "BUILD FAILED: main.pdf was not updated." >&2
  exit 1
fi

missing=$(tr -d '\n' < main.log | grep -o "Missing result [^ ]*" | sort -u | sed 's/Missing result //' || true)
undefined=$(grep -E "(Citation|Reference) .* undefined" main.log || true)
todos=$(grep -n '\\todo{' main.tex || true)
hand=$(grep -n -o '\\hand{[^}]*}' main.tex || true)
submit=$(grep -c '^\\booltrue{submit}' status.tex || true)
tracked=$(grep -c '^\\toggletrue{tracked}' status.tex || true)

echo "== $(pdfinfo main.pdf | grep Pages) | submit=$submit | change marks=$tracked"
echo "== missing numbers ($(printf '%s' "$missing" | grep -c . || true)):"; printf '%s\n' "$missing" | sed 's/^/  /'
echo "== \\todo left in main.tex ($(printf '%s' "$todos" | grep -c . || true)):"; printf '%s\n' "$todos" | cut -c1-120 | sed 's/^/  /'
echo "== hand-typed numbers to check:"; printf '%s\n' "$hand" | sed 's/^/  /'
[ -n "$undefined" ] && { echo "== undefined:"; echo "$undefined"; }
# One file to paste into Overleaf next to references.bib and neurips_2026.sty: main.tex with every \input inlined
python3 -I - <<'EOF'
import re
from pathlib import Path
inline = lambda match: Path(f"{match.group(1)}.tex").read_text()
text = re.sub(r"\\InputIfFileExists\{([^}]*)\}\{\}\{[^\n]*\}\n", inline, Path("main.tex").read_text())
Path("overleaf.tex").write_text(re.sub(r"\\input\{([^}]*)\}", inline, text))
EOF
echo "== overleaf.tex: $(wc -c < overleaf.tex) bytes, inputs left: $(grep -c '\\input{\|InputIfFileExists' overleaf.tex || true)"
if [ "$submit" -gt 0 ] && { [ -n "$missing" ] || [ -n "$todos" ] || [ -n "$undefined" ] || [ "$tracked" -gt 0 ]; }; then
  echo "NOT READY TO SUBMIT: turn off the blocks above in status.tex or fill them in." >&2
  exit 1
fi
