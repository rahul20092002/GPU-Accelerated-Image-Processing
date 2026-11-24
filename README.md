
# GPU-Accelerated Image Processing using CUDA

## Overview
This project implements and benchmarks three image-processing
operations using CUDA C++ and OpenCV:

1. Grayscale Conversion
2. Gaussian Blur (5x5)
3. Sobel Edge Detection

The objective is to explore GPU parallelism and compare
CUDA execution performance against CPU implementations.

## Hardware and Software
- GPU: NVIDIA Tesla T4
- CUDA Toolkit: 13.0
- OpenCV: 4.14.0
- Environment: Google Colab
- Programming Language: CUDA C++

## Benchmark Results

| Operation | CPU (ms) | GPU Kernel (ms) | GPU Total (ms) | Speedup |
|---|---:|---:|---:|---:|
| Grayscale | 17.7836 | 0.162906 | 5.02180 | 3.54x |
| Gaussian Blur | 634.9810 | 1.944860 | 8.50082 | 74.70x |
| Sobel | 103.8960 | 0.266669 | 2.50213 | 41.52x |

## Accuracy

| Operation | Maximum Pixel Difference |
|---|---:|
| Grayscale | 1 |
| Gaussian Blur | 0 |
| Sobel | 0 |

## Performance Analysis

- Gaussian Blur achieved the highest end-to-end speedup of 74.70x.
- Sobel Edge Detection achieved a speedup of 41.52x.
- Grayscale Conversion achieved a speedup of 3.54x.
- GPU total time includes host-to-device transfer, kernel execution,
  device-to-host transfer and synchronisation.
- GPU kernel time measures kernel execution separately.

## Project Structure

```text
GPU-Image-Processing/
├── src/
│   ├── benchmark_all.cu
│   ├── grayscale.cu
│   ├── gaussian_blur.cu
│   └── sobel.cu
├── data/
│   └── input.jpg
├── results/
│   ├── benchmark_results.csv
│   ├── graphs/
│   └── processed_images/
├── Makefile
├── run.sh
└── README.md
Benchmark Graphs
The results/graphs directory contains:
- CPU vs GPU execution time
- CPU vs GPU execution time (log scale)
- GPU speedup
- GPU execution-time breakdown
Execution
From the project root directory:
make
make run


Alternatively:
chmod +x run.sh
./run.sh


Conclusion
CUDA acceleration significantly improves the execution time
of computationally intensive image-processing operations.
Gaussian Blur and Sobel demonstrate particularly strong
parallel performance on the NVIDIA Tesla T4 GPU.
