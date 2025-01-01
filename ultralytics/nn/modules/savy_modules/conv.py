import torch.nn as nn
import torch
from modules.conv import Conv
from block import GlobalPoolingAggregator
from transformer import CrossAttnPE, VanillaAttnPE


class ElementWiseConvPool(nn.Module):
    def __init__(self, c1, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

        self.conv = nn.Conv2d(in_channels=c1, kernel_size=kernel_size)

    def forward(self, x1, x2):
        #Ensure the restored embeds equal
        assert x1.shape == x2.shape

        restored = x1 * x2
        weighted = self.conv(restored)
        unweighted = self.conv(x1)

        return weighted/unweighted
   

class SelfAttentivePooling(nn.Module):

    def __init__(self, c1, e_p=1, e_r=1.0, weighted_k=3, heads=2, prune=None):
        super().__init__()
       
        self.patch_size = e_p
        self.e_r = e_r
        self.c1 = c1
        reduced_dim = round(e_r * c1 / heads)

        if reduced_dim == 0: reduced_dim = 1

        embed_dim = reduced_dim * heads
        self.embed_dim = int(embed_dim)

        self.embedding = nn.Sequential(
            Conv(c1, (e_r * c1), e_p, e_p),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Unfold(kernel_size=e_p, padding=e_p, stride=e_p)
        )
        #Input shape --> (batch_size, sequence_length, embed_dimension)
        self.restore = nn.Sequential(
            #Pointwise Convolution
            nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c1),
            nn.Sigmoid()
        )

        #Element wise conv pool
        self.weighted_pool = ElementWiseConvPool(c1, weighted_k)
        self.attn = VanillaAttnPE(c1=embed_dim, num_heads=heads)
        self.prune = prune


    def forward(self, x):
        b,c,h,w = x.shape

        patches = self.embedding(x)
        attn_output = self.attn(patches) # (batch_size, patch_dim, num_patches) --> (batch_size, sequence_length, embed_dim)
       
        """ Need to check for size correction:

        batch_size, num_patches, embed_dim = attn_output.shape
        height, width = int(num_patches ** 0.5), int(num_patches ** 0.5)  # Assuming square patches
       
        attn_output = attn_output.transpose(1, 2)  # (batch_size, embed_dim, num_patches)
        attn_output = attn_output.view(batch_size, embed_dim, height, width) """

        restored = self.restore(attn_output)
        restored = torch.exp(restored)
        pooled = self.weighted_pool(restored, x)


        return pooled
   


class CrossAttentivePooling(SelfAttentivePooling):

    def __init__(self, c1, spatial_dim, c_final, e_p=1, e_r=1.0, weighted_k=3, heads=2):
        super().__init__(self, e_p=1, e_r=1.0, weighted_k=3)


        self.attn = CrossAttnPE(c1, num_heads=heads)
        self.c_final = c_final
        self.spatial_dim = spatial_dim
        self.aggregate = GlobalPoolingAggregator(c1, spatial_dim, c_final)
       
    def forward(self, x1, x2, x3):
        b,c,h,w = x1.shape

        q_patches = self.embedding(x1)
        k_patches = self.embedding(x2)
        v_patches = self.embedding(x3)
        attn_output = self.attn(q_patches, k_patches, v_patches) # (batch_size, patch_dim, num_patches) --> (batch_size, sequence_length, embed_dim)


        """ Check for size correction:
       
        batch_size, num_patches, embed_dim = attn_output.shape
        height, width = int(num_patches ** 0.5), int(num_patches ** 0.5)  # Assuming square patches
       
        attn_output = attn_output.transpose(1, 2)  # (batch_size, embed_dim, num_patches)
        attn_output = attn_output.view(batch_size, embed_dim, height, width) """


        restored = self.restore(attn_output)
        restored = torch.exp(restored)
        aggregated = self.aggregate(x1, x2, x3)

        #Utilize the Global Pooling Agregator for 1. Aggregated map comparison + feature (+ CBAM) with the FeatUp


        #Combine and project all the tensors into a global map
        pooled = self.weighted_pool(restored, aggregated)

        return pooled