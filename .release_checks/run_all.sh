#!/usr/bin/env bash
# ============================================================================
# GSSC-S2D2 release gate runner.
#
# Runs every check_*.py in this directory and prints a single progress line, so
# "how close is the release" has one number instead of an argument.
#
#   ./run_all.sh              run every gate, report, exit 1 if any fails
#   ./run_all.sh --selftest   run every gate's --selftest instead
#   ./run_all.sh -v           also echo each gate's own PASS/FAIL lines
#   ./run_all.sh check_asset  run only gates whose name matches a substring
#
# WHY A RUNNER AND NOT JUST pytest: these gates read artefacts that live OUTSIDE
# the repo (the 12 GB asset bundle, the built paper PDFs, the git object store,
# and the pinned tag). They are release-readiness measurements, not unit tests,
# and several take minutes. Keeping them out of pytest keeps CI honest about
# what it actually enforces -- one of the defects this harness exists for was
# CONTRIBUTING.md advertising five "CI-enforced" standards where CI ran two.
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

# Interpreter, roots and scratch are all env-overridable with portable defaults.  They were
# absolute paths into one machine once, which made the runner unusable anywhere else and
# published that machine's layout.  Every gate honours the same variables:
#
#   GSSC_PY           python to run the gates with   default: <repo>/.venv/bin/python, else python3
#   GSSC_REPO         checkout under test            default: the repo this script sits in
#   GSSC_ASSETS       asset staging bundle           default: <repo>/../GSSC-S2D2-assets
#   GSSC_PAPER        manuscript checkout            default: <repo>/../GSSC-paper
#   GSSC_EXPERIMENTS  internal experiments checkout  default: <repo>/../Semantic_Scene_Completion_LiDAR
#   TMPDIR            scratch root                   default: ~/.cache/gssc-release-checks
#
# The asset bundle, the manuscript and the experiments checkout are NOT part of the public
# release; the gates that read them cannot run without a copy, and say so rather than
# passing vacuously.  NEVER point TMPDIR at /tmp on the maintainer's box: a full /tmp has
# deadlocked it repeatedly.
PY="${GSSC_PY:-}"
if [[ -z "$PY" ]]; then
  for cand in "$REPO_ROOT/.venv/bin/python" "$(command -v python3 || true)"; do
    if [[ -n "$cand" && -x "$cand" ]]; then PY="$cand"; break; fi
  done
fi
# The interpreter is printed in the header below, deliberately: four gates import optional
# extras (PyMuPDF, safetensors, torch, tqdm) and a bare system python reports them as
# ERROR(1) with a ModuleNotFoundError, which looks like a broken gate rather than a wrong
# interpreter. If that happens, export GSSC_PY at the environment that has them.
if [[ -z "${TMPDIR:-}" ]]; then
  if [[ -n "${HOME:-}" ]]; then
    TMPDIR="$HOME/.cache/gssc-release-checks"
  else
    echo "set TMPDIR (or HOME): these gates must not scratch in /tmp" >&2; exit 2
  fi
fi
mkdir -p "$TMPDIR"
export TMPDIR

MODE="run"; VERBOSE=0; FILTER=""
for a in "$@"; do
  case "$a" in
    --selftest) MODE="selftest" ;;
    -v|--verbose) VERBOSE=1 ;;
    -*) echo "unknown flag: $a" >&2; exit 2 ;;
    *) FILTER="$a" ;;
  esac
done

if [[ ! -x "$PY" ]]; then
  echo "interpreter not found: $PY (override with GSSC_PY=...)" >&2; exit 2
fi

mapfile -t GATES < <(find "$HERE" -maxdepth 1 -name 'check_*.py' | sort)
if [[ -n "$FILTER" ]]; then
  mapfile -t GATES < <(printf '%s\n' "${GATES[@]}" | grep -- "$FILTER" || true)
fi
if [[ ${#GATES[@]} -eq 0 ]]; then
  echo "no gates found in $HERE${FILTER:+ matching '$FILTER'}" >&2; exit 2
fi

printf '\n%s\n' "── GSSC-S2D2 release gates ─────────────────────────────────────────────"
printf '%s\n'   "   mode=$MODE  gates=${#GATES[@]}  repo=$REPO_ROOT@$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
printf '%s\n'   "   py=$PY  TMPDIR=$TMPDIR"

# IMPORT PROBE. Printing the interpreter is not enough on its own: four gates import optional
# extras, and without them they report FAIL or die with a ModuleNotFoundError that reads as a
# broken gate rather than a wrong python. Measured 2026-08-23 -- a bare /usr/local/bin/python
# turned a 16/16 board into "7 failing" with nothing on screen connecting that to the
# interpreter. Say it before the gates run, not after.
_missing=$("$PY" - <<'PYPROBE' 2>/dev/null
mods = ("fitz", "huggingface_hub", "safetensors", "torch", "yaml")
out = []
for m in mods:
    try:
        __import__(m)
    except Exception:
        out.append(m)
print(",".join(out))
PYPROBE
)
if [[ -n "$_missing" ]]; then
  printf '%s\n' "   !! this interpreter cannot import: ${_missing//,/, }"
  printf '%s\n' "      Gates that need them will report failures that are NOT defects."
  printf '%s\n' "      Use the project venv, or: GSSC_PY=<interpreter> $0 $*"
fi
printf '%s\n\n' "────────────────────────────────────────────────────────────────────────"

GREEN=(); RED=(); BROKE=()
for g in "${GATES[@]}"; do
  name="$(basename "$g" .py)"
  start=$SECONDS
  if [[ "$MODE" == "selftest" ]]; then
    out="$("$PY" "$g" --selftest 2>&1)"; rc=$?
  else
    out="$("$PY" "$g" 2>&1)"; rc=$?
  fi
  dur=$(( SECONDS - start ))

  # rc 0 = green, rc 1 = the gate did its job and found a defect, anything else
  # = the gate itself is broken. Conflating "found a problem" with "crashed" is
  # how a harness quietly stops measuring, so they are reported separately.
  if [[ $rc -eq 0 ]]; then
    GREEN+=("$name"); mark="PASS"
  elif [[ $rc -eq 1 ]]; then
    RED+=("$name");   mark="FAIL"
  else
    BROKE+=("$name"); mark="ERROR($rc)"
  fi

  printf '  %-6s %-34s %3ds\n' "$mark" "$name" "$dur"
  if [[ $VERBOSE -eq 1 || $rc -ne 0 ]]; then
    printf '%s\n' "$out" | sed 's/^/           /'
    printf '\n'
  fi
done

total=${#GATES[@]}; green=${#GREEN[@]}; red=${#RED[@]}; broke=${#BROKE[@]}
printf '\n%s\n' "────────────────────────────────────────────────────────────────────────"
printf '   %d/%d green' "$green" "$total"
[[ $red   -gt 0 ]] && printf '   ·   %d failing: %s' "$red" "$(printf '%s ' "${RED[@]}")"
[[ $broke -gt 0 ]] && printf '   ·   %d BROKEN: %s' "$broke" "$(printf '%s ' "${BROKE[@]}")"
printf '\n%s\n\n' "────────────────────────────────────────────────────────────────────────"

# A broken gate is worse than a failing one: it means we have stopped measuring
# something. Surface it with a distinct exit code so a caller can tell apart
# "the release is not ready" from "the harness is not working".
[[ $broke -gt 0 ]] && exit 2
[[ $red   -gt 0 ]] && exit 1
exit 0
