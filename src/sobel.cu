
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <algorithm>

using namespace std;

#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl; \
        return 1; \
    } \
}

// Sobel X and Y kernels
const int h_Gx[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

const int h_Gy[9] = {
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
};

__constant__ int d_Gx[9];
__constant__ int d_Gy[9];

// CPU Sobel implementation
void cpuSobel(
    const cv::Mat& input,
    cv::Mat& output
) {

    int width = input.cols;
    int height = input.rows;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            int gx = 0;
            int gy = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {

                    int px = min(max(x + kx, 0), width - 1);
                    int py = min(max(y + ky, 0), height - 1);

                    int pixel = input.at<unsigned char>(py, px);

                    int k = (ky + 1) * 3 + (kx + 1);

                    gx += pixel * h_Gx[k];
                    gy += pixel * h_Gy[k];
                }
            }

            int magnitude = static_cast<int>(
                sqrtf(static_cast<float>(gx * gx + gy * gy))
            );

            output.at<unsigned char>(y, x) =
                static_cast<unsigned char>(min(magnitude, 255));
        }
    }
}

// GPU Sobel kernel
__global__ void gpuSobel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int gx = 0;
    int gy = 0;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {

            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);

            int pixel = input[py * width + px];

            int k = (ky + 1) * 3 + (kx + 1);

            gx += pixel * d_Gx[k];
            gy += pixel * d_Gy[k];
        }
    }

    float magnitude = sqrtf(
        static_cast<float>(gx * gx + gy * gy)
    );

    output[y * width + x] =
        static_cast<unsigned char>(
            fminf(magnitude, 255.0f)
        );
}

int main() {

    cv::Mat color = cv::imread("input.jpg");

    if (color.empty()) {
        cerr << "Could not load input.jpg" << endl;
        return 1;
    }

    // Convert input image to grayscale
    cv::Mat gray;
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);

    int width = gray.cols;
    int height = gray.rows;

    size_t imageSize = width * height;

    cv::Mat cpuOutput(height, width, CV_8UC1);
    cv::Mat gpuOutput(height, width, CV_8UC1);

    // Copy Sobel coefficients to GPU constant memory
    CHECK(cudaMemcpyToSymbol(d_Gx, h_Gx, sizeof(h_Gx)));
    CHECK(cudaMemcpyToSymbol(d_Gy, h_Gy, sizeof(h_Gy)));

    unsigned char *d_input, *d_output;

    CHECK(cudaMalloc(&d_input, imageSize));
    CHECK(cudaMalloc(&d_output, imageSize));

    dim3 threads(16, 16);

    dim3 blocks(
        (width + 15) / 16,
        (height + 15) / 16
    );

    // CPU benchmark
    auto cpuStart = chrono::high_resolution_clock::now();

    cpuSobel(gray, cpuOutput);

    auto cpuEnd = chrono::high_resolution_clock::now();

    double cpuTime =
        chrono::duration<double, milli>(
            cpuEnd - cpuStart
        ).count();

    // GPU benchmark
    auto gpuStart = chrono::high_resolution_clock::now();

    CHECK(cudaMemcpy(
        d_input, gray.data, imageSize,
        cudaMemcpyHostToDevice
    ));

    gpuSobel<<<blocks, threads>>>(
        d_input, d_output, width, height
    );

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(
        gpuOutput.data, d_output, imageSize,
        cudaMemcpyDeviceToHost
    ));

    auto gpuEnd = chrono::high_resolution_clock::now();

    double gpuTime =
        chrono::duration<double, milli>(
            gpuEnd - gpuStart
        ).count();

    // Compare results
    double difference = cv::norm(
        cpuOutput, gpuOutput, cv::NORM_INF
    );

    cv::imwrite("sobel_cpu.jpg", cpuOutput);
    cv::imwrite("sobel_gpu.jpg", gpuOutput);

    cout << "\n========== SOBEL RESULTS ==========\n";

    cout << "Image dimensions: "
         << width << " x " << height << endl;

    cout << "CPU time: " << cpuTime << " ms" << endl;

    cout << "GPU total time: " << gpuTime << " ms" << endl;

    cout << "Speedup: "
         << cpuTime / gpuTime << "x" << endl;

    cout << "Maximum pixel difference: "
         << difference << endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
