#include <cassert>
#include <torch/torch.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <iostream>

using torch::Tensor;

Tensor create_block_matrix(const Tensor & kernel, int input_size, int output_size) {
    int I = kernel.size(3);  // Kernel height (I)
    int J = kernel.size(4);   // Kernel width (J)
    int t = output_size;  // Output size (from sliding window dimensions)
    int n = input_size;  // Input size (from input dimensions)

    std::vector<Tensor> K_blocks; // Initialize a vector to store the block matrices

    // Construct the block matrix
    for (int i = 0; i < I; i++) {

        auto K_i = torch::zeros({t, n}); // Initialize the block matrix K_i
        
        for (int row = 0; row < I; row++) {  
            // Extract the row of the kernel (the sliding window)
            auto kernel_row = kernel[i];  // This is a row from the kernel (of size J)

            // Now slide the kernel row into the K_i matrix
            // The starting position for this kernel row will depend on the row of K_i
            int start_idx = row;
            int end_idx = start_idx + (J-row);  // Kernel width is J

            // Assign the kernel values to the correct positions in K_i
            K_i.index({row, torch::indexing::Slice(start_idx, end_idx)}) = kernel_row;
        }
        K_blocks.push_back(K_i);
    }

    // Concatenate the block matrices to form the block matrix M
    Tensor M = K_blocks[0];  // Start with the first K_i matrix

    for (size_t i = 1; i < K_blocks.size(); i++) {
        M = torch::cat({M, K_blocks[i]}, 1);
    }

    return M;
}

namespace adaptive_conv {
    // Custom Forward Pass for Adaptive Convolution
    Tensor forward(Tensor input, Tensor filters) {
        int batch_size = input.size(0);
        int channels = input.size(1);
        int height = input.size(2);
        int width = input.size(3);

        int kernel_height = filters.size(3);
        int kernel_width = filters.size(4);

        int output_height = height - kernel_height + 1;
        int output_width = width - kernel_width + 1;

        // Initialize an output tensor with proper dimensions
        Tensor output = torch::zeros({batch_size, channels, output_height, output_width});

        // Loop over each batch element (image) independently
        for (int b = 0; b < batch_size; b++) {
            // Loop over each channel independently
            for (int c = 0; c < channels; c++) {
                // Extract the kernel for the current channel
                auto filter = filters[b][c]; // [I, J] kernel for the current channel

                // Create the block matrix for the current kernel
                Tensor M = create_block_matrix(filter, height, output_height);

                // Flatten the input tensor spatially for the current batch and channel
                auto input_flat = input[b][c].view({-1}); // [n*n]

                // Perform matrix multiplication: M * input_flat
                auto result = torch::matmul(M, input_flat);

                // Reshape the result to the output dimensions
                output[b][c] = result.view({output_height, output_width});
            }
        }
        return output;
    }


// Custom Gradient for Input (Backward Pass)
Tensor grad_input(Tensor grad_output, Tensor filters) {
    auto B = grad_output.sizes()[0];
    auto C_out = grad_output.sizes()[1];
    auto H_out = grad_output.sizes()[2];
    auto W_out = grad_output.sizes()[3];


    assert(filters.sizes()[0] == B);
    assert(filters.sizes()[1] == H_out);
    assert(filters.sizes()[2] == W_out);
    auto I = filters.sizes()[3];
    auto J = filters.sizes()[4];
    assert(I == J);


    auto H_in = H_out + I - 1;
    auto W_in = W_out + J - 1;


    assert(grad_output.dtype() == filters.dtype());


    auto out = torch::zeros({B, C_out, H_in, W_in }, grad_output.dtype());


    // Unfold operations
    // Iterate over the batch
    for (int b = 0; b < B; b++) {
        // Unroll grad_output into a row vector
        auto grad_output_unrolled = grad_output[b].view({C_out, H_out * W_out});


        // Prepare the sparse filter matrix
        auto filter_sparse = torch::zeros({H_out * W_out, H_in * W_in});
        for (int h_out = 0; h_out < H_out; h_out++) {
            for (int w_out = 0; w_out < W_out; w_out++) {
                for (int i = 0; i < I; i++) {
                    for (int j = 0; j < J; j++) {
                        int h_in = h_out + i;
                        int w_in = w_out + j;
                        filter_sparse[h_out * W_out + w_out][h_in * W_in + w_in] = filters[b][h_out][w_out][i][j];
                    }
                }
            }
        }


        // Perform sparse matrix multiplication
        auto grad_input_flat = torch::matmul(filter_sparse, grad_output_unrolled);


        // Reshape back to (B, C_out, H_in, W_in)
        out[b] = grad_input_flat.view({B, C_out, H_in, W_in});
    }


    return out;
}


// Custom Gradient for Filters (Backward Pass)
Tensor grad_filters(Tensor grad_output, Tensor input) {


    auto B = grad_output.sizes()[0];
    auto C_out = grad_output.sizes()[1];
    auto H_out = grad_output.sizes()[2];
    auto W_out = grad_output.sizes()[3];


    assert(input.sizes()[0] == B);
    auto C_in = input.sizes()[1];
    auto H_in = input.sizes()[2];
    auto W_in = input.sizes()[3];


    assert(H_in > H_out);
    assert(W_in > W_out);


    auto I = W_in - W_out + 1;
    auto J = H_in - H_out + 1;
   
    assert(grad_output.dtype() == input.dtype());


    // Output tensor for gradients of filters
    auto out = torch::zeros({B, H_out, W_out, I, J}, grad_output.dtype());


    for (int b = 0; b < B; b++) {
        // Unroll input tensor into a matrix
        auto unfolded_input = torch::zeros({C_in * I * J, H_out * W_out});
        for (int h_out = 0; h_out < H_out; h_out++) {
            for (int w_out = 0; w_out < W_out; w_out++) {
                for (int i = 0; i < I; i++) {
                    for (int j = 0; j < J; j++) {
                        int h_in = h_out + i;
                        int w_in = w_out + j;


                        // Fill unfolded_input with the corresponding input values
                        for (int c = 0; c < C_in; c++) {
                            unfolded_input[c * I * J + i * J + j][h_out * W_out + w_out] =
                                input[b][c][h_in][w_in];
                        }
                    }
                }
            }
        }


        // Unroll grad_output into a row-major matrix
        auto grad_output_unrolled = grad_output[b].view({C_out, H_out * W_out});


        // Perform matrix multiplication
        auto grad_filter_flat = torch::matmul(grad_output_unrolled, unfolded_input.t());


        // Reshape grad_filter_flat back into (C_out, I, J)
        auto grad_filter_reshaped = grad_filter_flat.view({C_out, I, J});


        // Assign to the grad_filters tensor
        out[b] = grad_filter_reshaped;
        }


        return out;


    }
}


// Expose the functions to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &adaptive_conv::forward, "Adaptive Convolution Forward");
    m.def("grad_input", &adaptive_conv::grad_input, "Adaptive Convolution Gradient Input");
    m.def("grad_filters", &adaptive_conv::grad_filters, "Adaptive Convolution Gradient Filters");
}