#include "sobel_kernel.cuh"

namespace {

constexpr int kBlockDimX = 16;
constexpr int kBlockDimY = 16;

__constant__ int kSobelGx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1},
};

__constant__ int kSobelGy[3][3] = {
    {-1, -2, -1},
    {0, 0, 0},
    {1, 2, 1},
};

__device__ inline int ClampInt(int value, int lo, int hi) {
  return value < lo ? lo : (value > hi ? hi : value);
}

__global__ void SobelKernel(const unsigned char* input, unsigned char* output,
                             int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  int gx = 0;
  int gy = 0;

  // 3x3 convolution with edge-clamped boundary handling.
#pragma unroll
  for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
    for (int dx = -1; dx <= 1; ++dx) {
      const int sample_x = ClampInt(x + dx, 0, width - 1);
      const int sample_y = ClampInt(y + dy, 0, height - 1);
      const unsigned char pixel = input[sample_y * width + sample_x];
      gx += kSobelGx[dy + 1][dx + 1] * pixel;
      gy += kSobelGy[dy + 1][dx + 1] * pixel;
    }
  }

  const int magnitude = static_cast<int>(sqrtf(static_cast<float>(gx * gx + gy * gy)));
  output[y * width + x] = static_cast<unsigned char>(ClampInt(magnitude, 0, 255));
}

}  // namespace

void RunSobelKernel(const unsigned char* d_input, unsigned char* d_output,
                     int width, int height, cudaStream_t stream) {
  const dim3 block_dim(kBlockDimX, kBlockDimY);
  const dim3 grid_dim((width + kBlockDimX - 1) / kBlockDimX,
                       (height + kBlockDimY - 1) / kBlockDimY);
  SobelKernel<<<grid_dim, block_dim, 0, stream>>>(d_input, d_output, width,
                                                   height);
}