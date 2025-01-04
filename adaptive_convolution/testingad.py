import torch
from conv_runner import AdaptiveConv  # Assuming your module is defined here

def test_adaptive_conv():
    # Define the dimensions
    B = 2  # Batch size
    C_in = 3  # Input channels
    H_in, W_in = 32, 32  # Input height and width
    C_out = 4  # Output channels
    I, J = 3, 3  # Filter size
    stride = 1  # Stride (example)

    # Calculate output dimensions
    H_out = (H_in - I) // stride + 1
    W_out = (W_in - J) // stride + 1

    # Create random tensors for input and filters
    input_tensor = torch.randn(B, C_in, H_in, W_in, requires_grad=True, dtype=torch.double)  # Input
    filters = torch.randn(B, C_out, H_out, I, J, requires_grad=True, dtype=torch.double)  # Adaptive filters

    # Forward pass using AdaptiveConv.apply()
    output = AdaptiveConv.apply(input_tensor, filters)

    # Validate output dimensions
    assert output.shape == (B, C_out, H_out, W_out), f"Unexpected output shape: {output.shape}"

    # Print results for manual inspection
    print("Input tensor shape:", input_tensor.shape)
    print("Filters shape:", filters.shape)
    print("Output tensor shape:", output.shape)

    # Test backward pass using autograd.gradcheck
    gradcheck_input = (input_tensor.clone().detach().requires_grad_(True),
                       filters.clone().detach().requires_grad_(True))

    # Ensure AdaptiveConv supports double precision for gradcheck
    assert torch.autograd.gradcheck(AdaptiveConv.apply, gradcheck_input), "Gradcheck failed!"

    print("Backward pass successful, gradients computed, gradcheck passed.")

# Run the test
if __name__ == "__main__":
    test_adaptive_conv()
