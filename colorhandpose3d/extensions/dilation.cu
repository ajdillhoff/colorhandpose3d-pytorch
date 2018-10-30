#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void dilation_cuda_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ kernel,
        scalar_t* __restrict__ output,
        int input_rows,
        int input_cols,
        int kernel_rows,
        int kernel_cols,
        int output_rows,
        int output_cols,
        int stride_rows,
        int stride_cols,
        int rate_rows,
        int rate_cols,
        int pad_top,
        int pad_left) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z * blockDim.z + threadIdx.z;

    const int h_beg = y * stride_rows - pad_top;
    const int w_beg = x * stride_cols - pad_left;

    auto cur_val = -99999.0f;
    for (int h = 0; h < kernel_rows; h++) {
        const int h_in = h_beg + h * rate_rows;
        if (h_in >= 0 && h_in < input_rows) {
            for (int w = 0; w < kernel_cols; w++) {
                const int w_in = w_beg + w * rate_cols;
                if (w_in >= 0 && w_in < input_cols) {
                    auto val =
                        input[batch * (input_rows * input_cols) + h_in * input_cols + w_in]
                            + kernel[h * kernel_cols + w];
                    if (val > cur_val) {
                        cur_val = val;
                    }
                }
            }
        }
    }
    if (cur_val < -1) {
        printf("blockIdx.x %d, blockDim.x %d, threadIdx.x %d\n", blockIdx.x, blockDim.x, threadIdx.x);
        printf("blockIdx.y %d, blockDim.y %d, threadIdx.y %d\n", blockIdx.y, blockDim.y, threadIdx.y);
        printf("blockIdx.z %d, blockDim.z %d, threadIdx.z %d\n", blockIdx.z, blockDim.z, threadIdx.z);
        printf("h_beg %d, w_beg %d\n", h_beg, w_beg);
    }
    output[batch * (output_rows * output_cols) + y * output_cols + x] = cur_val;
}

at::Tensor dilation_cuda(
        at::Tensor input,
        at::Tensor kernel,
        int stride_rows,
        int stride_cols,
        int rate_rows,
        int rate_cols,
        int pad_top,
        int pad_left,
        int output_height,
        int output_width) {

    const auto batch_size = input.size(0);
    const auto output_size = output_height * output_width;

    const int input_rows = input.size(1);
    const int input_cols = input.size(2);
    const int kernel_rows = kernel.size(0);
    const int kernel_cols = kernel.size(1);

    const dim3 threads(8, 8, 1);
    const dim3 blocks((output_height + threads.x - 1) / threads.x, (output_width + threads.y - 1) / threads.y, batch_size);

    auto output = at::zeros(input.type(), {batch_size, output_height, output_width});
    std::cout << blocks.x << " " << blocks.y << " " << blocks.z << std::endl;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "dilation_cuda", ([&] {
      dilation_cuda_kernel<scalar_t><<<blocks, threads>>>(
              input.data<scalar_t>(),
              kernel.data<scalar_t>(),
              output.data<scalar_t>(),
              input_rows,
              input_cols,
              kernel_rows,
              kernel_cols,
              output_height,
              output_width,
              stride_rows,
              stride_cols,
              rate_rows,
              rate_cols,
              pad_top,
              pad_left);
    }));

    cudaDeviceSynchronize();

    return output;
}
