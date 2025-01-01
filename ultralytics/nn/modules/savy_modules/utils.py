import torch


def channel_prune(x: torch.Tensor, prune: float = 0.1, regrow: float=0.2) -> torch.Tensor:
    m, n, h, w = x.shape
    channels = int(n*prune)


    weights = x.view(m, n* h * w)
    weights_partitioned = torch.split(weights, h * w, dim=1)


    f_norms = [torch.norm(mat, p='fro') for mat in weights_partitioned]
    sorted_indices = torch.argsort(torch.tensor(f_norms))
    pruned_channels = sorted_indices[:channels]

   
    weights_pruned = weights.clone()  
    for i in pruned_channels:
        weights_pruned[:, i, :, :] = 0  
   
    # Optionally, compute normalized momentum (importance) and re-grow channels
    # Normalized momentum (based on F-norms of the weights)
    normalized_momentum = torch.abs(torch.tensor(f_norms)) / torch.sum(torch.abs(torch.tensor(f_norms)))
   
    # Step 7: Re-grow channels based on importance (if above the threshold)
    weights_regrown = weights.clone()  # Clone to regenerate channels
    for i in pruned_channels:
        if normalized_momentum[i] > regrow:  # If channel is important enough, regrow
            weights_regrown[:, i, :, :] = weights[:, i, :, :]  # Re-enable the channel
   
    return weights_pruned, weights_regrown