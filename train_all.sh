#!/bin/bash
# Train on all ~1.1M CGOS games in chunks of 60k, 12 epochs each.
# 18 chunks of 60k = 1,080,000 + 1 chunk of 20k = 1,100,000 total.
# Starts from AI/models/v3.pth and saves AI/models/full_c01.pth … full_c19.pth.

set -e
cd "$(dirname "$0")"
source venv/bin/activate

GAMES_DIR="data/Games/"
MODELS_DIR="AI/models"
DEVICE="mps"
EPOCHS=12
BATCH=512
LR=1e-3

RESUME="AI/models/v3.pth"

run_chunk() {
  local idx=$1
  local offset=$2
  local size=$3
  local out="$MODELS_DIR/full_c$(printf '%02d' $idx).pth"

  echo ""
  echo "========================================="
  echo "Chunk $idx/19 — offset=$offset, games=$size"
  echo "Resume: $RESUME → Out: $out"
  echo "========================================="

  python -u -m AI.training.supervised \
    --games     "$GAMES_DIR" \
    --offset    $offset \
    --max-games $size \
    --resume    "$RESUME" \
    --epochs    $EPOCHS \
    --batch     $BATCH \
    --lr        $LR \
    --device    $DEVICE \
    --out       "$out"

  RESUME="$out"
}

run_chunk  1       0  60000
run_chunk  2   60000  60000
run_chunk  3  120000  60000
run_chunk  4  180000  60000
run_chunk  5  240000  60000
run_chunk  6  300000  60000
run_chunk  7  360000  60000
run_chunk  8  420000  60000
run_chunk  9  480000  60000
run_chunk 10  540000  60000
run_chunk 11  600000  60000
run_chunk 12  660000  60000
run_chunk 13  720000  60000
run_chunk 14  780000  60000
run_chunk 15  840000  60000
run_chunk 16  900000  60000
run_chunk 17  960000  60000
run_chunk 18 1020000  60000
run_chunk 19 1080000  20000

echo ""
echo "All 1.1M games done. Final model: $RESUME"
