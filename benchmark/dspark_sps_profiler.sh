#!/bin/bash
# =============================================================================
# dspark_sps_profiler.sh
#
# One-shot generator for a DSpark SPS (Steps Per Second) cost table.
#
# It launches a DSpark inference server configured for SPS profiling
# (static ragged-verify + step-time recording + simulate_acc_len=1.0),
# waits until it is healthy, runs `dspark_sps_profiler run` to collect
# per-step records, then `dspark_sps_profiler fit` to fit the SPS JSON,
# and finally shuts the server down (trap-backed cleanup on any exit path).
#
# Usage:
#   MODEL_PATH=... DRAFT_MODEL_PATH=... ./dspark_sps_profiler.sh [--help]
#
# All knobs are configurable via environment variables (defaults below).
# Set FRACS="0.25 0.5 0.75 1.0" to switch to the off-diagonal 2D table mode
# (the server is then launched with SGLANG_RAGGED_VERIFY_MODE=compact).
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
# --- Required ---
MODEL_PATH="${MODEL_PATH:-}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-}"

# --- Server ---
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
BASE_URL="${BASE_URL:-http://localhost:${PORT}}"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-7}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"

# --- Profiler: run ---
OUT="${OUT:-dspark_sps.json}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-256}"
BATCH_SIZES="${BATCH_SIZES:-}"          # optional explicit sweep, e.g. "32 64 128"
REPEATS="${REPEATS:-1}"
FRACS="${FRACS:-}"                      # optional off-diagonal sweep, e.g. "0.25 0.5 1.0"
STEADY_STEPS="${STEADY_STEPS:-32}"
MIN_STEADY_SECONDS="${MIN_STEADY_SECONDS:-10.0}"
ROUND_TIMEOUT="${ROUND_TIMEOUT:-300}"
INPUT_LEN="${INPUT_LEN:-16}"
TEMPERATURE="${TEMPERATURE:-1.0}"

# --- Flow control ---
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
READY_POLL_INTERVAL="${READY_POLL_INTERVAL:-5}"
SERVER_LOG="${SERVER_LOG:-dspark_sps_server.log}"
PYTHON="${PYTHON:-python3}"

SERVER_PID=""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
usage() {
    cat <<'EOF'
Usage: MODEL_PATH=... DRAFT_MODEL_PATH=... ./dspark_sps_profiler.sh

Required environment variables:
  MODEL_PATH          Path/name of the target (verify) model
  DRAFT_MODEL_PATH    Path/name of the DSpark draft checkpoint

Common overrides:
  PORT, TP_SIZE, MEM_FRACTION_STATIC, DSPARK_BLOCK_SIZE, EXTRA_SERVER_ARGS
  OUT, MAX_BATCH_SIZE, BATCH_SIZES, REPEATS, FRACS,
  STEADY_STEPS, MIN_STEADY_SECONDS, ROUND_TIMEOUT, INPUT_LEN, TEMPERATURE
  SERVER_READY_TIMEOUT, SERVER_LOG, PYTHON

Example (off-diagonal 2D table):
  MODEL_PATH=/models/kimi-k3 DRAFT_MODEL_PATH=/models/kimi-k3-dspark \
  TP_SIZE=8 FRACS="0.25 0.5 0.75 1.0" ./dspark_sps_profiler.sh
EOF
}

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
    log "ERROR: $*" >&2
    exit 1
}

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Shutting down DSpark server (pid=$SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        # Give it a moment to drain, then force-kill if needed.
        for _ in $(seq 1 30); do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            log "Server did not exit gracefully; sending SIGKILL."
            kill -9 "$SERVER_PID" 2>/dev/null || true
            wait "$SERVER_PID" 2>/dev/null || true
        else
            wait "$SERVER_PID" 2>/dev/null || true
        fi
        log "Server stopped."
    fi
    SERVER_PID=""
}

wait_for_server() {
    local deadline=$((SECONDS + SERVER_READY_TIMEOUT))
    local code=""
    log "Waiting for server at ${BASE_URL}/health (timeout=${SERVER_READY_TIMEOUT}s)..."
    while (( SECONDS < deadline )); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            die "Server process exited before becoming healthy; see tail of ${SERVER_LOG}:"
        fi
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "${BASE_URL}/health" 2>/dev/null || true)
        if [[ "$code" == "200" ]]; then
            log "Server is healthy."
            return 0
        fi
        sleep "$READY_POLL_INTERVAL"
    done
    log "Server did not become healthy within ${SERVER_READY_TIMEOUT}s; last HTTP code: '${code}'." >&2
    log "--- Tail of ${SERVER_LOG} ---" >&2
    tail -n 50 "$SERVER_LOG" >&2 2>/dev/null || true
    return 1
}

# ---------------------------------------------------------------------------
# Validate inputs & export the SPS-profiling environment
# ---------------------------------------------------------------------------
[[ -z "$MODEL_PATH" ]] && die "MODEL_PATH is required (see --help)."
[[ -z "$DRAFT_MODEL_PATH" ]] && die "DRAFT_MODEL_PATH is required (see --help)."

if [[ -n "$FRACS" ]]; then
    log "FRACS set -> off-diagonal 2D table mode; launching server in compact ragged-verify mode."
    export SGLANG_RAGGED_VERIFY_MODE="compact"
else
    export SGLANG_RAGGED_VERIFY_MODE="static"
fi
export SGLANG_DSPARK_ENABLE_SPS_RECORD="1"
export SGLANG_SIMULATE_ACC_LEN="1.0"

# ---------------------------------------------------------------------------
# Launch the server
# ---------------------------------------------------------------------------
SERVER_ARGS=(
    --model-path "$MODEL_PATH"
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "$DRAFT_MODEL_PATH"
    --speculative-dspark-block-size "$DSPARK_BLOCK_SIZE"
    --tp-size "$TP_SIZE"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --host "$HOST"
    --port "$PORT"
)
# shellcheck disable=SC2086
read -r -a EXTRA_ARGS <<< "$EXTRA_SERVER_ARGS"
if (( ${#EXTRA_ARGS[@]} > 0 )); then
    SERVER_ARGS+=("${EXTRA_ARGS[@]}")
fi

log "Launching DSpark server (model=$MODEL_PATH, draft=$DRAFT_MODEL_PATH, tp=$TP_SIZE, port=$PORT)..."
log "Command: $PYTHON -m sglang.launch_server ${SERVER_ARGS[*]}"

nohup "$PYTHON" -m sglang.launch_server "${SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap cleanup EXIT INT TERM

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    die "Failed to start the server; see ${SERVER_LOG}."
fi

wait_for_server || exit 1

# ---------------------------------------------------------------------------
# Collect (run) and fit
# ---------------------------------------------------------------------------
RUN_ARGS=(
    --base-url "$BASE_URL"
    --out "$OUT"
    --repeats "$REPEATS"
    --min-steady-steps "$STEADY_STEPS"
    --min-steady-seconds "$MIN_STEADY_SECONDS"
    --round-timeout "$ROUND_TIMEOUT"
    --input-len "$INPUT_LEN"
    --temperature "$TEMPERATURE"
)
if [[ -n "$BATCH_SIZES" ]]; then
    read -r -a BS_ARGS <<< "$BATCH_SIZES"
    RUN_ARGS+=(--batch-size "${BS_ARGS[@]}")
else
    RUN_ARGS+=(--max-batch-size "$MAX_BATCH_SIZE")
fi
if [[ -n "$FRACS" ]]; then
    read -r -a FRAC_ARGS <<< "$FRACS"
    RUN_ARGS+=(--fracs "${FRAC_ARGS[@]}")
fi

log "Collecting SPS records: $PYTHON -m sglang.benchmark.dspark_sps_profiler run ${RUN_ARGS[*]}"
"$PYTHON" -m sglang.benchmark.dspark_sps_profiler run "${RUN_ARGS[@]}"

log "Fitting SPS table: $PYTHON -m sglang.benchmark.dspark_sps_profiler fit --out $OUT --plot --self-check"
"$PYTHON" -m sglang.benchmark.dspark_sps_profiler fit --out "$OUT" --plot --self-check

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "SPS table generated successfully. Artifacts:"
for artifact in \
    "$OUT" \
    "${OUT%.json}.records.jsonl" \
    "${OUT%.json}.rounds.jsonl" \
    "$OUT.manifest.json" \
    "${OUT%.json}.plot.png"; do
    if [[ -f "$artifact" ]]; then
        log "  - $artifact"
    fi
done
log "Done."
