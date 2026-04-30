#!/bin/bash
# Train on 500k games in chunks of 40k, each starting from the previous checkpoint.
# 12 chunks of 40k = 480k, then 1 chunk of 20k = 500k total.
# Each chunk: 5 epochs. Total: 65 passes across 500k unique games.

set -e
cd "$(dirname "$0")"
source venv/bin/activate

GAMES_DIR="AI/training/data/Games/"
MODELS_DIR="AI/models"
DEVICE="mps"
CHUNK=40000
EPOCHS=5
BATCH=512
LR=1e-3

RESUME="AI/models/v2.pth"

run_chunk() {
  local idx=$1
  local offset=$2
  local size=$3
  local out="$MODELS_DIR/supervised_c$(printf '%02d' $idx).pth"

  echo ""
  echo "========================================="
  echo "Chunk $idx — offset=$offset, games=$size"
  echo "Resume: $RESUME → Out: $out"
  echo "========================================="

  python -u -m AI.training.supervised \
    --games   "$GAMES_DIR" \
    --offset  $offset \
    --max-games $size \
    --resume  "$RESUME" \
    --epochs  $EPOCHS \
    --batch   $BATCH \
    --lr      $LR \
    --device  $DEVICE \
    --out     "$out"

  RESUME="$out"
}

run_chunk  1      0  40000
run_chunk  2  40000  40000
run_chunk  3  80000  40000
run_chunk  4 120000  40000
run_chunk  5 160000  40000
run_chunk  6 200000  40000
run_chunk  7 240000  40000
run_chunk  8 280000  40000
run_chunk  9 320000  40000
run_chunk 10 360000  40000
run_chunk 11 400000  40000
run_chunk 12 440000  40000
run_chunk 13 480000  20000

echo ""
echo "All 500k games done. Final model: $RESUME"
