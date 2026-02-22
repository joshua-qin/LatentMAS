#!/usr/bin/env bash
# Run only TextMAS sequential, 20 samples on GSM8K (Qwen3-4B). Small batch size to avoid OOM.
# Usage: OUT_DIR=./gsm8k_20_evals ./run_textmas_20_only.sh

set -e
MODEL="${MODEL:-Qwen/Qwen3-4B}"
OUT_DIR="${OUT_DIR:-./gsm8k_20_evals}"
GENERATE_BS="${GENERATE_BS:-4}"
mkdir -p "$OUT_DIR"

# Free GPU: kill any existing run.py
pkill -f "run.py" 2>/dev/null || true
sleep 2

LOG_FILE="$OUT_DIR/gsm8k_textmas_sequential_20.jsonl"

echo "=== TextMAS sequential — 20 samples GSM8K ($MODEL), generate_bs=$GENERATE_BS ==="
python run.py --method text_mas --model_name "$MODEL" --task gsm8k --prompt sequential \
  --max_samples 20 --max_new_tokens 2048 --generate_bs "$GENERATE_BS" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "Done. Results in $OUT_DIR/gsm8k_textmas_sequential_20.jsonl (last line = JSON summary)"
