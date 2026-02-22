#!/usr/bin/env bash
# Run 20 evals each: one-shot (baseline), sequential TextMAS, sequential LatentMAS on GSM8K.
# Uses standard HF backend. Set MODEL and OUT_DIR as needed.

set -e
MODEL="${MODEL:-Qwen/Qwen3-4B}"
OUT_DIR="${OUT_DIR:-./gsm8k_20_evals}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
mkdir -p "$OUT_DIR"

if [ "$SKIP_BASELINE" = "0" ]; then
  echo "=== 1/3 One-shot (baseline) — 20 samples GSM8K ==="
  python run.py --method baseline --model_name "$MODEL" --task gsm8k --max_samples 20 --max_new_tokens 2048 \
    | tee "$OUT_DIR/gsm8k_baseline_20.jsonl"
  echo ""
fi

echo "=== Sequential TextMAS — 20 samples GSM8K ==="
python run.py --method text_mas --model_name "$MODEL" --task gsm8k --prompt sequential --max_samples 20 --max_new_tokens 2048 \
  | tee "$OUT_DIR/gsm8k_textmas_sequential_20.jsonl"

echo ""
echo "=== Sequential LatentMAS — 20 samples GSM8K ==="
python run.py --method latent_mas --model_name "$MODEL" --task gsm8k --prompt sequential --max_samples 20 --max_new_tokens 2048 \
  | tee "$OUT_DIR/gsm8k_latentmas_sequential_20.jsonl"

echo ""
echo "Done. Results (accuracy line) saved under $OUT_DIR/"
