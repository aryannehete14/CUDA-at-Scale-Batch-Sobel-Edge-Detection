#!/usr/bin/env bash
# run.sh — builds the project, generates sample data (if data/ is empty),
# and runs the batch edge-detection program, producing a proof-of-execution
# log in logs/run.log.
#
# Usage: ./run.sh [num_images] [image_size]

set -e

NUM_IMAGES="${1:-200}"
IMAGE_SIZE="${2:-256}"

echo "== Building project =="
make clean
make all

if [ -z "$(ls -A data 2>/dev/null)" ]; then
  echo "== data/ is empty; generating ${NUM_IMAGES} synthetic ${IMAGE_SIZE}x${IMAGE_SIZE} images =="
  python3 generate_data.py --count "${NUM_IMAGES}" --size "${IMAGE_SIZE}" --out data
else
  echo "== Using existing images in data/ =="
fi

mkdir -p output logs

echo "== Running batch edge detection on GPU =="
./bin/edge_detect --input data --output output --log logs/run.log

echo "== Done. Log: logs/run.log | Output images: output/ =="
tail -n 8 logs/run.log