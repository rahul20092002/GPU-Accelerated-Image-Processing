
#!/bin/bash

set -e

cd "$(dirname "$0")"

echo "GPU Image Processing - CUDA"

mkdir -p results/processed_images

make clean
make run
