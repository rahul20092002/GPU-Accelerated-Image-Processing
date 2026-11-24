
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl; \
        return 1; \
    } \
}

// 5x5 Gaussian kernel
__constant__ float gaussianKernel[25];

const float h_kernel[25] = {
    1,  4,  6,  4,  1,
    4, 16, 24, 16,  4,
    6, 24, 36, 24,  6,
    4, 16, 24, 16,  4,
    1,  4,  6,  4,  1
};

// CPU Gaussian blur
void cpuGaussianBlur(
    const cv::Mat& input,
    cv::Mat& output
) {
    int width = input.cols;
    int height = input.rows;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            for (int c = 0; c < 3; c++) {

                float sum = 0.0f;

                for (int ky = -2; ky <= 2; ky++) {
                    for (int kx = -2; kx <= 2; kx++) {

                        int px = min(max(x + kx, 0), width - 1);
                        int py = min(max(y + ky, 0), height - 1);

                        float pixel =
                            input.ptr<unsigned char>(py)[px * 3 + c];

                        float weight =
                            h_kernel[(ky + 2) * 5 + (kx + 2)];

                        sum += pixel * weight;
                    }
                }

                output.ptr<unsigned char>(y)[x * 3 + c] =
                    static_cast<unsigned char>(sum / 256.0f + 0.5f);
            }
        }
    }
}

// GPU Gaussian blur kernel
__global__ void gpuGaussianBlur(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    for (int c = 0; c < 3; c++) {

        float sum = 0.0f;

        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {

                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);

                int idx = (py * width + px) * 3 + c;

                float pixel = input[idx];

                float weight =
                    gaussianKernel[(ky + 2) * 5 + (kx + 2)];

                sum += pixel * weight;
            }
        }

        output[(y * width + x) * 3 + c] =
            static_cast<unsigned char>(sum / 256.0f + 0.5f);
    }
}

int main() {

    cv::Mat input = cv::imread("input.jpg");

    if (input.empty()) {
        cerr << "Could not load input.jpg" << endl;
        return 1;
    }

    int width = input.cols;
    int height = input.rows;

    size_t imageSize = width * height * 3;

    cv::Mat cpuOutput(height, width, CV_8UC3);
    cv::Mat gpuOutput(height, width, CV_8UC3);

    // Copy Gaussian weights to GPU constant memory
    CHECK(cudaMemcpyToSymbol(
        gaussianKernel,
        h_kernel,
        sizeof(h_kernel)
    ));

    // Allocate GPU memory
    unsigned char *d_input, *d_output;

    CHECK(cudaMalloc(&d_input, imageSize));
    CHECK(cudaMalloc(&d_output, imageSize));

    dim3 threads(16, 16);

    dim3 blocks(
        (width + 15) / 16,
        (height + 15) / 16
    );

    // CPU execution
    auto cpuStart = chrono::high_resolution_clock::now();

    cpuGaussianBlur(input, cpuOutput);

    auto cpuEnd = chrono::high_resolution_clock::now();

    double cpuTime =
        chrono::duration<double, milli>(
            cpuEnd - cpuStart
        ).count();

    // GPU execution
    auto gpuStart = chrono::high_resolution_clock::now();

    CHECK(cudaMemcpy(
        d_input, input.data, imageSize,
        cudaMemcpyHostToDevice
    ));

    gpuGaussianBlur<<<blocks, threads>>>(
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

    // Compare CPU and GPU results
    double difference = cv::norm(
        cpuOutput, gpuOutput, cv::NORM_INF
    );

    cv::imwrite("gaussian_cpu.jpg", cpuOutput);
    cv::imwrite("gaussian_gpu.jpg", gpuOutput);

    cout << "\n========== GAUSSIAN BLUR RESULTS ==========\n";

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
