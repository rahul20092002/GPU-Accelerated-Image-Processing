
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace std::chrono;

#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl; \
        return 1; \
    } \
}

const int N = 5;

// Gaussian kernel
const float h_gaussian[25] = {
    1,4,6,4,1,
    4,16,24,16,4,
    6,24,36,24,6,
    4,16,24,16,4,
    1,4,6,4,1
};

__constant__ float d_gaussian[25];

// Sobel kernels
const int h_Gx[9] = {
    -1,0,1,
    -2,0,2,
    -1,0,1
};

const int h_Gy[9] = {
    -1,-2,-1,
     0, 0, 0,
     1, 2, 1
};

__constant__ int d_Gx[9];
__constant__ int d_Gy[9];

// =====================================================
// GRAYSCALE
// =====================================================

void cpuGray(const cv::Mat& input, cv::Mat& output) {

    for (int y = 0; y < input.rows; y++) {

        const uchar* src = input.ptr<uchar>(y);
        uchar* dst = output.ptr<uchar>(y);

        for (int x = 0; x < input.cols; x++) {

            int i = x * 3;

            float value =
                0.299f * src[i + 2] +
                0.587f * src[i + 1] +
                0.114f * src[i];

            dst[x] = static_cast<uchar>(value);
        }
    }
}

__global__ void gpuGray(
    const uchar* input, uchar* output,
    int width, int height
) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {

        int i = (y * width + x) * 3;

        output[y * width + x] =
            0.299f * input[i + 2] +
            0.587f * input[i + 1] +
            0.114f * input[i];
    }
}

// =====================================================
// GAUSSIAN BLUR
// =====================================================

void cpuGaussian(
    const cv::Mat& input, cv::Mat& output
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
                            input.ptr<uchar>(py)[px * 3 + c];

                        sum += pixel *
                            h_gaussian[(ky + 2) * 5 + kx + 2];
                    }
                }

                output.ptr<uchar>(y)[x * 3 + c] =
                    static_cast<uchar>(sum / 256.0f + 0.5f);
            }
        }
    }
}

__global__ void gpuGaussian(
    const uchar* input, uchar* output,
    int width, int height
) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    for (int c = 0; c < 3; c++) {

        float sum = 0.0f;

        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {

                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);

                int idx = (py * width + px) * 3 + c;

                sum += input[idx] *
                    d_gaussian[(ky + 2) * 5 + kx + 2];
            }
        }

        output[(y * width + x) * 3 + c] =
            static_cast<uchar>(sum / 256.0f + 0.5f);
    }
}

// =====================================================
// SOBEL EDGE DETECTION
// =====================================================

void cpuSobel(
    const cv::Mat& input, cv::Mat& output
) {

    int width = input.cols;
    int height = input.rows;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            int gx = 0, gy = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {

                    int px = min(max(x + kx, 0), width - 1);
                    int py = min(max(y + ky, 0), height - 1);

                    int pixel = input.ptr<uchar>(py)[px];

                    int k = (ky + 1) * 3 + kx + 1;

                    gx += pixel * h_Gx[k];
                    gy += pixel * h_Gy[k];
                }
            }

            int magnitude = static_cast<int>(
                sqrtf(static_cast<float>(gx * gx + gy * gy))
            );

            output.ptr<uchar>(y)[x] =
                static_cast<uchar>(min(magnitude, 255));
        }
    }
}

__global__ void gpuSobel(
    const uchar* input, uchar* output,
    int width, int height
) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int gx = 0, gy = 0;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {

            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);

            int pixel = input[py * width + px];

            int k = (ky + 1) * 3 + kx + 1;

            gx += pixel * d_Gx[k];
            gy += pixel * d_Gy[k];
        }
    }

    float magnitude = sqrtf(
        static_cast<float>(gx * gx + gy * gy)
    );

    output[y * width + x] =
        static_cast<uchar>(fminf(magnitude, 255.0f));
}

// =====================================================
// BENCHMARK HELPER
// =====================================================

struct Result {
    string name;
    double cpu;
    double kernel;
    double total;
    double speedup;
    double difference;
};

template <typename CPUFunc, typename GPUFunc>
Result benchmark(
    string name,
    const cv::Mat& input,
    cv::Mat& cpuOutput,
    cv::Mat& gpuOutput,
    CPUFunc cpuFunc,
    GPUFunc gpuFunc,
    size_t inputSize,
    size_t outputSize,
    int width,
    int height,
    bool color
) {

    unsigned char *d_input, *d_output;

    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);

    dim3 threads(16,16);
    dim3 blocks(
        (width + 15) / 16,
        (height + 15) / 16
    );

    // Warm-up
    cudaMemcpy(d_input, input.data, inputSize,
               cudaMemcpyHostToDevice);

    gpuFunc<<<blocks, threads>>>(
        d_input, d_output, width, height
    );

    cudaDeviceSynchronize();

    // CPU timing
    auto cpuStart = high_resolution_clock::now();

    for (int i = 0; i < N; i++) {
        cpuFunc(input, cpuOutput);
    }

    auto cpuEnd = high_resolution_clock::now();

    double cpuTime =
        duration<double, milli>(cpuEnd - cpuStart).count() / N;

    // GPU kernel-only timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < N; i++) {
        gpuFunc<<<blocks, threads>>>(
            d_input, d_output, width, height
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernelTotal = 0;
    cudaEventElapsedTime(&kernelTotal, start, stop);

    double kernelTime = kernelTotal / N;

    // GPU end-to-end timing
    auto gpuStart = high_resolution_clock::now();

    for (int i = 0; i < N; i++) {

        cudaMemcpy(d_input, input.data, inputSize,
                   cudaMemcpyHostToDevice);

        gpuFunc<<<blocks, threads>>>(
            d_input, d_output, width, height
        );

        cudaDeviceSynchronize();

        cudaMemcpy(gpuOutput.data, d_output, outputSize,
                   cudaMemcpyDeviceToHost);
    }

    auto gpuEnd = high_resolution_clock::now();

    double totalTime =
        duration<double, milli>(gpuEnd - gpuStart).count() / N;

    double difference = cv::norm(
        cpuOutput, gpuOutput, cv::NORM_INF
    );

    Result result = {
        name,
        cpuTime,
        kernelTime,
        totalTime,
        cpuTime / totalTime,
        difference
    };

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);

    return result;
}

// =====================================================
// MAIN
// =====================================================

int main() {

    cv::Mat color = cv::imread("data/input.jpg");

    if (color.empty()) {
        cerr << "input.jpg not found!" << endl;
        return 1;
    }

    int width = color.cols;
    int height = color.rows;

    cout << "Image: " << width << " x " << height << endl;
    cout << "Iterations: " << N << endl;

    cv::Mat gray(height, width, CV_8UC1);
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);

    cv::Mat cpuGrayOut(height, width, CV_8UC1);
    cv::Mat gpuGrayOut(height, width, CV_8UC1);

    cv::Mat cpuGaussianOut(height, width, CV_8UC3);
    cv::Mat gpuGaussianOut(height, width, CV_8UC3);

    cv::Mat cpuSobelOut(height, width, CV_8UC1);
    cv::Mat gpuSobelOut(height, width, CV_8UC1);

    cudaMemcpyToSymbol(
        d_gaussian, h_gaussian, sizeof(h_gaussian)
    );

    cudaMemcpyToSymbol(d_Gx, h_Gx, sizeof(h_Gx));
    cudaMemcpyToSymbol(d_Gy, h_Gy, sizeof(h_Gy));

    vector<Result> results;

    results.push_back(benchmark(
        "Grayscale", color, cpuGrayOut, gpuGrayOut,
        cpuGray, gpuGray,
        color.total() * color.elemSize(),
        gray.total() * gray.elemSize(),
        width, height, true
    ));

    results.push_back(benchmark(
        "Gaussian Blur", color,
        cpuGaussianOut, gpuGaussianOut,
        cpuGaussian, gpuGaussian,
        color.total() * color.elemSize(),
        color.total() * color.elemSize(),
        width, height, true
    ));

    results.push_back(benchmark(
        "Sobel", gray, cpuSobelOut, gpuSobelOut,
        cpuSobel, gpuSobel,
        gray.total() * gray.elemSize(),
        gray.total() * gray.elemSize(),
        width, height, false
    ));

    // Save outputs
    cv::imwrite("results/processed_images/results_gray.jpg", gpuGrayOut);
    cv::imwrite("results/processed_images/results_gaussian.jpg", gpuGaussianOut);
    cv::imwrite("results/processed_images/results_sobel.jpg", gpuSobelOut);

    // Save CSV
    ofstream csv("results/benchmark_results.csv");

    csv << "Operation,CPU_ms,GPU_Kernel_ms,GPU_Total_ms,"
           "Speedup,Max_Pixel_Difference\n";

    cout << "\n========== CONSOLIDATED RESULTS ==========\n";

    for (const auto& r : results) {

        cout << "\n" << r.name << endl;
        cout << "CPU: " << r.cpu << " ms" << endl;
        cout << "GPU kernel: " << r.kernel << " ms" << endl;
        cout << "GPU total: " << r.total << " ms" << endl;
        cout << "Speedup: " << r.speedup << "x" << endl;
        cout << "Max difference: " << r.difference << endl;

        csv << r.name << ","
            << r.cpu << ","
            << r.kernel << ","
            << r.total << ","
            << r.speedup << ","
            << r.difference << "\n";
    }

    csv.close();

    cout << "\nCSV saved as benchmark_results.csv" << endl;

    return 0;
}
