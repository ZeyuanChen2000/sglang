#!/bin/bash
# =============================================================================
# dspark_sts_fit.sh
#
# One-shot generator for a DSpark confidence STS (Sequential Temperature
# Scaling) calibration JSON.
#
# It launches a DSpark inference server with a confidence-head draft
# checkpoint (non-static ragged-verify; identity temperatures are enforced
# because no --speculative-dspark-confidence-sts-path is passed), runs a
# representative workload while SGLANG_DSPARK_STS_COLLECT_PATH is set so the
# server dumps raw confidence logits + prefix masks into .pt shards, then
# runs `sglang.benchmark.dspark_sts_fit` to fit the per-position temperatures,
# and finally shuts the server down (trap-backed cleanup on any exit path).
#
# Usage:
#   MODEL_PATH=... DRAFT_MODEL_PATH=... ./dspark_sts_fit.sh [--help]
#
# All knobs are configurable via environment variables (defaults below).
# The workload is a random-input `sglang.benchmark.serving` run by default;
# set LOAD_CMD to any shell command (e.g. your own eval script) to collect
# shards from a representative deployment workload.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
# --- Required ---
MODEL_PATH="${MODEL_PATH:-}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-}"    # MUST be a DSpark draft checkpoint with a confidence head

# --- Server ---
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
BASE_URL="${BASE_URL:-http://localhost:${PORT}}"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-7}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"

# --- STS collection ---
# Path *stem* (no ".pt") where the server dumps shards:
#   <COLLECT_DIR>/<COLLECT_STEM>.0.pt, .1.pt, ...
COLLECT_DIR="${COLLECT_DIR:-sts_shards}"
COLLECT_STEM="${COLLECT_STEM:-shards_stem}"
SGLANG_DSPARK_STS_COLLECT_PATH="${COLLECT_DIR}/${COLLECT_STEM}"

# --- Representative workload (default: random serving benchmark) ---
LOAD_CMD="${LOAD_CMD:-}"
NUM_PROMPTS="${NUM_PROMPTS:-2000}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-128}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-128}"
REQUEST_RATE="${REQUEST_RATE:-20.0}"

# --- Fit ---
OUT="${OUT:-sts_calibration.json}"
GAMMA="${GAMMA:-}"                          # optional; if empty, inferred from shards
NUM_BINS="${NUM_BINS:-15}"

# --- Flow control ---
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
READY_POLL_INTERVAL="${READY_POLL_INTERVAL:-5}"
SERVER_LOG="${SERVER_LOG:-dspark_sts_server.log}"
PYTHON="${PYTHON:-python3}"

SERVER_PID=""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
usage() {
    cat <<'EOF'
Usage: MODEL_PATH=... DRAFT_MODEL_PATH=... ./dspark_sts_fit.sh

Required environment variables:
  MODEL_PATH          Path/name of the target (verify) model
  DRAFT_MODEL_PATH    Path/name of the DSpark draft checkpoint (MUST have a
                      confidence head; otherwise no STS shards are collected)

Common overrides:
  PORT, TP_SIZE, MEM_FRACTION_STATIC, DSPARK_BLOCK_SIZE, EXTRA_SERVER_ARGS
  COLLECT_DIR, COLLECT_STEM
  LOAD_CMD, NUM_PROMPTS, RANDOM_INPUT_LEN, RANDOM_OUTPUT_LEN, REQUEST_RATE
  OUT, GAMMA, NUM_BINS
  SERVER_READY_TIMEOUT, SERVER_LOG, PYTHON

Notes:
  - Do NOT pass --speculative-dspark-confidence-sts-path via
    EXTRA_SERVER_ARGS: collection requires identity temperatures and the
    server will refuse to start with both set.
  - The server must run in non-static ragged-verify mode (the default); a
    static-mode/folded verify step skips STS collection.

Example with a custom workload (e.g. your GSM8K eval script):
  MODEL_PATH=/models/kimi-k3 DRAFT_MODEL_PATH=/models/kimi-k3-dspark \
  TP_SIZE=8 GAMMA=7 \
  LOAD_CMD="python /path/to/your_eval.py --base-url http://localhost:30000" \
  ./dspark_sts_fit.sh
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
# Validate inputs & export the STS-collection environment
# ---------------------------------------------------------------------------
[[ -z "$MODEL_PATH" ]] && die "MODEL_PATH is required (see --help)."
[[ -z "$DRAFT_MODEL_PATH" ]] && die "DRAFT_MODEL_PATH is required (see --help)."

if grep -q -- '--speculative-dspark-confidence-sts-path' <<< "$EXTRA_SERVER_ARGS"; then
    die "STS collection requires identity temperatures: do NOT pass " \
        "--speculative-dspark-confidence-sts-path via EXTRA_SERVER_ARGS."
fi

# Collection environment. IMPORTANT: do NOT set SGLANG_RAGGED_VERIFY_MODE
# (must stay non-static) and do NOT set SGLANG_SIMULATE_ACC_LEN=1.0
# (that would defeat spec acceptance and make the collected correct-prefix
# masks meaningless).
mkdir -p "$COLLECT_DIR"
export SGLANG_DSPARK_STS_COLLECT_PATH

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
# Run the representative workload (dumps .pt shards while it runs)
# ---------------------------------------------------------------------------
if [[ -n "$LOAD_CMD" ]]; then
    log "Running custom workload: $LOAD_CMD"
    bash -c "$LOAD_CMD"
else
    log "Running default random workload: num_prompts=$NUM_PROMPTS, input_len=$RANDOM_INPUT_LEN, output_len=$RANDOM_OUTPUT_LEN, rate=$REQUEST_RATE"
    "$PYTHON" -m sglang.benchmark.serving \
        --backend sglang \
        --model "$MODEL_PATH" \
        --base-url "$BASE_URL" \
        --dataset-name random \
        --num-prompts "$NUM_PROMPTS" \
        --random-input-len "$RANDOM_INPUT_LEN" \
        --random-output-len "$RANDOM_OUTPUT_LEN" \
        --request-rate "$REQUEST_RATE"
fi

# ---------------------------------------------------------------------------
# Sanity-check the collected shards, then fit
# ---------------------------------------------------------------------------
SHARD_COUNT=$(ls "${SGLANG_DSPARK_STS_COLLECT_PATH}".*.pt 2>/dev/null | wc -l)
if [[ "$SHARD_COUNT" -eq 0 ]]; then
    die "No STS shards collected at ${SGLANG_DSPARK_STS_COLLECT_PATH}.*.pt. " \
        "Check that the draft checkpoint carries a confidence head and that " \
        "the server runs in non-static ragged-verify mode; see ${SERVER_LOG}."
fi
log "Collected $SHARD_COUNT STS shard(s) under ${COLLECT_DIR}/."

FIT_ARGS=(--data-glob "${SGLANG_DSPARK_STS_COLLECT_PATH}.*.pt" --out "$OUT")
if [[ -n "$GAMMA" ]]; then
    FIT_ARGS+=(--gamma "$GAMMA")
else
    log "GAMMA not set; fit will infer gamma from the shards. Make sure the " \
        "inferred gamma matches the gamma you deploy with."
fi
FIT_ARGS+=(--num-bins "$NUM_BINS")

log "Fitting STS temperatures: $PYTHON -m sglang.benchmark.dspark_sts_fit ${FIT_ARGS[*]}"
"$PYTHON" -m sglang.benchmark.dspark_sts_fit "${FIT_ARGS[@]}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "STS calibration generated successfully. Artifacts:"
for artifact in \
    "$OUT" \
    "${COLLECT_DIR}/"*.pt; do
    if [[ -f "$artifact" ]]; then
        log "  - $artifact"
    fi
done
log "Deploy with: --speculative-dspark-confidence-sts-path $(realpath "$OUT")"
log "Done."
