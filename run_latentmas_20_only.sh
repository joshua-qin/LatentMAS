#!/usr/bin/env bash
# Run only LatentMAS, 20 samples on GSM8K (Qwen3-4B). Uses smaller batch size to avoid OOM.
# Usage: LATENT_STEPS=20 REALIGN=1 OUT_DIR=./gsm8k_20_evals ./run_latentmas_20_only.sh

set -e
MODEL="${MODEL:-Qwen/Qwen3-4B}"
OUT_DIR="${OUT_DIR:-./gsm8k_20_evals}"
LATENT_STEPS="${LATENT_STEPS:-0}"
REALIGN="${REALIGN:-0}"
mkdir -p "$OUT_DIR"

# Kill any existing run.py so GPU is free
pkill -f "run.py" 2>/dev/null || true
sleep 2

if [ "$LATENT_STEPS" = "0" ]; then
  BASE="gsm8k_latentmas_sequential_20"
else
  BASE="gsm8k_latentmas_sequential_20_steps${LATENT_STEPS}"
fi
[ "$REALIGN" = "1" ] && BASE="${BASE}_realign"
LOG_FILE="$OUT_DIR/${BASE}.jsonl"

REALIGN_FLAG=""
[ "$REALIGN" = "1" ] && REALIGN_FLAG="--latent_space_realign"

echo "=== LatentMAS sequential — 20 samples GSM8K ($MODEL), latent_steps=$LATENT_STEPS, realign=$REALIGN ==="
python run.py --method latent_mas --model_name "$MODEL" --task gsm8k --prompt sequential \
  --max_samples 20 --max_new_tokens 2048 --generate_bs 4 --latent_steps "$LATENT_STEPS" \
  $REALIGN_FLAG \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "Done. Results in $OUT_DIR (file: $(basename "$LOG_FILE"), last line = JSON summary)"
