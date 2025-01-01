#include <cassert>
#include <torch/torch.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

using torch::Tensor;


namespace adaptive_conv {
// Custom Forward Pass for Adaptive Convolution
Tensor adaptive_conv_forward(Tensor input, Tensor filters) {
    // Get shapes of input and filters
    auto B = input.sizes()[0];
    auto C_in = input.sizes()[1];
    auto H_in = input.sizes()[2];
    auto W_in = input.sizes()[3];


    assert(filters.sizes()[0] == B);
    auto H_out = filters.sizes()[1];
    auto W_out = filters.sizes()[2];
    auto I = filters.sizes()[3];
    auto J = filters.sizes()[4];


    assert(I == J);
    assert(H_out + I - 1 == H_in);
    assert(W_out + J - 1 == W_in);


    // Reshape input to be a vector of patches (unfolded image)
    Tensor image_vector = torch::flatten(input, 1);  // Shape: (B, C_in * H_in * W_in)


    // Reshape filter to a sparse matrix form (with zeros added for padding/stride simulation)
    auto unfolded_filters = torch::flatten(filters, 1);  // Shape: (B, H_out * W_out * C_in * I * J)


    // Prepare the filter to be sparse by adding zeros
    auto sparse_filter = torch::zeros_like(unfolded_filters);  // Start with a zeroed matrix


    // Manually insert non-zero filter values into the sparse matrix
    for (int b = 0; b < B; ++b) {
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; ++w_out) {
                for (int c_in = 0; c_in < C_in; ++c_in) {
                    for (int i = 0; i < I; ++i) {
                        for (int j = 0; j < J; ++j) {
                            int row_idx = b * (H_out * W_out * C_in * I * J) + (h_out * W_out * C_in * I * J) + (w_out * C_in * I * J) + (c_in * I * J) + (i * J) + j;
                            sparse_filter[row_idx] = unfolded_filters[b * (H_out * W_out * C_in * I * J) + (h_out * W_out * C_in * I * J) + (w_out * C_in * I * J) + (c_in * I * J) + (i * J) + j];
                        }
                    }
                }
            }
        }
    }


    // Sparse matmul
    auto unfolded_input = image_vector.view({B, C_in * H_in * W_in, -1});  // Shape: (B, C_in * H_in * W_in, H_out * W_out)
    auto out = torch::matmul(sparse_filter, unfolded_input);


    // Reshape the output to (B, C_in, H_out, W_out)
    out = out.view({B, C_in, H_out, W_out}).permute({0, 3, 1, 2});  // Change to (B, C_in, H_out, W_out)


    return out;
}


// Custom Gradient for Input (Backward Pass)
Tensor adaptive_conv_grad_input(Tensor grad_output, Tensor filters) {
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
Tensor adaptive_conv_grad_filters(Tensor grad_output, Tensor input) {


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
    m.def("forward", &adaptive_conv::adaptive_conv_forward, "Adaptive Convolution Forward");
    m.def("grad_input", &adaptive_conv::adaptive_conv_grad_input, "Adaptive Convolution Gradient Input");
    m.def("grad_filters", &adaptive_conv::adaptive_conv_grad_filters, "Adaptive Convolution Gradient Filters");
}
