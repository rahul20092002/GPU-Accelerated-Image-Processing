
#include <iostream>
#include <cuda_runtime.h>

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

    const int width = 4;
    const int height = 4;

    unsigned char h_input[width * height * 3];

    for (int i = 0; i < width * height; i++) {
        h_input[i * 3] = 255;
        h_input[i * 3 + 1] = 0;
        h_input[i * 3 + 2] = 0;
    }

    unsigned char h_output[width * height];

    unsigned char *d_input, *d_output;

    cudaMalloc(&d_input, sizeof(h_input));
    cudaMalloc(&d_output, sizeof(h_output));

    cudaMemcpy(d_input, h_input, sizeof(h_input),
               cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16,
                (height + 15) / 16);

    rgbToGrayscale<<<blocks, threads>>>(
        d_input, d_output, width, height
    );

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, sizeof(h_output),
               cudaMemcpyDeviceToHost);

    std::cout << "Grayscale Output:\n";

    for (int i = 0; i < width * height; i++) {
        std::cout << (int)h_output[i] << " ";
        if ((i + 1) % width == 0)
            std::cout << "\n";
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
