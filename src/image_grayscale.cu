
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << cudaGetErrorString(err) << std::endl; \
        return 1; \
    } \
}

__global__ void rgbToGrayscale(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;

        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];

        output[y * width + x] =
            0.299f * r + 0.587f * g + 0.114f * b;
    }
}

int main() {

    cv::Mat image = cv::imread("input.jpg");

    if (image.empty()) {
        std::cerr << "Could not load image!" << std::endl;
        return 1;
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    int width = image.cols;
    int height = image.rows;

    size_t inputSize = width * height * 3;
    size_t outputSize = width * height;

    unsigned char *d_input, *d_output;

    CHECK(cudaMalloc(&d_input, inputSize));
    CHECK(cudaMalloc(&d_output, outputSize));

    CHECK(cudaMemcpy(d_input, image.data, inputSize,
                     cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16,
                (height + 15) / 16);

    rgbToGrayscale<<<blocks, threads>>>(
        d_input, d_output, width, height
    );

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cv::Mat gray(height, width, CV_8UC1);

    CHECK(cudaMemcpy(gray.data, d_output, outputSize,
                     cudaMemcpyDeviceToHost));

    cv::imwrite("grayscale_output.jpg", gray);

    std::cout << "Image processed successfully!" << std::endl;
    std::cout << "Image dimensions: "
              << width << " x " << height << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
