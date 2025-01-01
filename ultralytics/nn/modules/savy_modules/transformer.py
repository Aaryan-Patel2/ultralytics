from modules.transformer import AIFI



class VanillaAttnPE(AIFI):

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0):
        super().__init__(c1, cm=cm, num_heads=num_heads, dropout=dropout)
   
    # Noted as the operation for computing the bask attention embeds; no use of dropout or normalization layers
    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        embeds = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        return embeds

    def forward(self, x):
        b, c, h, w = x.shape
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        embeds = self.forward_pre(x, pos=pos_embed)
       
        return embeds


class CrossAttnPE(VanillaAttnPE):
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0):
        super().__init__(c1, cm=cm, num_heads=num_heads, dropout=dropout)
   
    def forward_pre(self, src1, src2, src3,src_mask=None, src_key_padding_mask=None, pos=None):
            q = self.with_pos_embed(src1, pos)
            k = self.with_pos_embed(src2, pos)
            v = self.with_pos_embed(src3, pos)

            embeds = self.ma(q, k, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
           
            return embeds

    def forward(self, x1, x2, x3):
        b, c, h, w = x1.shape
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        embeds = self.forward_pre(x1, x2, x3, pos=pos_embed)
       
        return embeds
