
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl; \
        return 1; \
    } \
}

// CPU grayscale implementation
void cpuGrayscale(const cv::Mat& input, cv::Mat& output) {

    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {

            cv::Vec3b pixel = input.at<cv::Vec3b>(y, x);

            unsigned char b = pixel[0];
            unsigned char g = pixel[1];
            unsigned char r = pixel[2];

            output.at<unsigned char>(y, x) =
                static_cast<unsigned char>(
                    0.299f * r + 0.587f * g + 0.114f * b
                );
        }
    }
}

// CUDA grayscale kernel
__global__ void gpuGrayscale(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {

        int idx = (y * width + x) * 3;

        unsigned char b = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char r = input[idx + 2];

        output[y * width + x] =
            0.299f * r + 0.587f * g + 0.114f * b;
    }
}

int main() {

    cv::Mat input = cv::imread("input.jpg");

    if (input.empty()) {
        cerr << "Image not found!" << endl;
        return 1;
    }

    int width = input.cols;
    int height = input.rows;

    size_t inputSize = width * height * 3;
    size_t outputSize = width * height;

    cv::Mat cpuOutput(height, width, CV_8UC1);
    cv::Mat gpuOutput(height, width, CV_8UC1);

    const int iterations = 100;

    // ---------------- CPU BENCHMARK ----------------

    auto cpuStart = high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        cpuGrayscale(input, cpuOutput);
    }

    auto cpuEnd = high_resolution_clock::now();

    double cpuTime =
        duration<double, milli>(cpuEnd - cpuStart).count()
        / iterations;

    // ---------------- GPU MEMORY ALLOCATION ----------------

    unsigned char *d_input, *d_output;

    CHECK(cudaMalloc(&d_input, inputSize));
    CHECK(cudaMalloc(&d_output, outputSize));

    dim3 threads(16, 16);
    dim3 blocks(
        (width + 15) / 16,
        (height + 15) / 16
    );

    // Warm-up launch
    CHECK(cudaMemcpy(d_input, input.data, inputSize,
                     cudaMemcpyHostToDevice));

    gpuGrayscale<<<blocks, threads>>>(
        d_input, d_output, width, height
    );

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // ---------------- GPU KERNEL-ONLY TIME ----------------

    cudaEvent_t start, stop;

    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {
        gpuGrayscale<<<blocks, threads>>>(
            d_input, d_output, width, height
        );
    }

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float kernelTotal = 0;

    CHECK(cudaEventElapsedTime(
        &kernelTotal, start, stop
    ));

    double kernelTime = kernelTotal / iterations;

    // ---------------- GPU TOTAL TIME ----------------

    auto gpuStart = high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {

        CHECK(cudaMemcpy(d_input, input.data, inputSize,
                         cudaMemcpyHostToDevice));

        gpuGrayscale<<<blocks, threads>>>(
            d_input, d_output, width, height
        );

        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(gpuOutput.data, d_output, outputSize,
                         cudaMemcpyDeviceToHost));
    }

    auto gpuEnd = high_resolution_clock::now();

    double gpuTotalTime =
        duration<double, milli>(gpuEnd - gpuStart).count()
        / iterations;

    // ---------------- RESULTS ----------------

    double speedupKernel = cpuTime / kernelTime;
    double speedupTotal = cpuTime / gpuTotalTime;

    double difference = cv::norm(
        cpuOutput, gpuOutput, cv::NORM_INF
    );

    cout << "\n========== BENCHMARK RESULTS ==========\n";

    cout << "Image dimensions: "
         << width << " x " << height << endl;

    cout << "Iterations: " << iterations << endl;

    cout << "\nCPU average time: "
         << cpuTime << " ms" << endl;

    cout << "GPU kernel average time: "
         << kernelTime << " ms" << endl;

    cout << "GPU total average time: "
         << gpuTotalTime << " ms" << endl;

    cout << "\nKernel-only speedup: "
         << speedupKernel << "x" << endl;

    cout << "End-to-end speedup: "
         << speedupTotal << "x" << endl;

    cout << "\nMaximum pixel difference: "
         << difference << endl;

    cv::imwrite("benchmark_output.jpg", gpuOutput);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
