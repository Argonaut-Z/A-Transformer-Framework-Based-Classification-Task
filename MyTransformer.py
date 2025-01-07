import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_

is_print_shape = False

def multi_head_attention_forward(query,     # [tgt_len, batch_size, embed_dim]
                                 key,       # [src_len, batch_size, embed_dim]
                                 value,     # [src_len, batch_size, embed_dim]
                                 num_heads,
                                 dropout_p,
                                 out_proj,  # [embed_dim = v_dim * num_heads, embed_dim = v_dim * num_heads]
                                 training=True,
                                 key_padding_mask=None,  # [batch_size, tgt_len/src_len]
                                 q_proj=None,   # [embed_dim, q_dim * num_heads]
                                 k_proj=None,   # [embed_dim, k_dim * num_heads]
                                 v_proj=None,   # [embed_dim, v_dim * num_heads]
                                 attn_mask=None,    # [tgt_len, src_len] or [num_heads * batch_size, tgt_len, src_len]
                                 ):
    
    q = q_proj(query)
    # [tgt_len, batch_size, embed_dim] × [embed_dim, q_dim * num_heads] = [tgt_len, batch_size, q_dim * num_heads]

    k = k_proj(key)
    # [src_len, batch_size, embed_dim] × [embed_dim, q_dim * num_heads] = [src_len, batch_size, q_dim * num_heads]
    
    v = v_proj(value)
    # [src_len, batch_size, embed_dim] × [embed_dim, q_dim * num_heads] = [src_len, batch_size, q_dim * num_heads]
    
    tgt_len, bsz, embed_dim = query.size()  # [tgt_len, batch_size, embed_dim]
    src_len = key.size(0)
    head_dim = embed_dim // num_heads   # num_heads * head_dim = embed_dim, q_dim = k_dim = v_dim = head_dim
    scaling = float(head_dim) ** -0.5
    q = q * scaling   # [tgt_len, batch_size, q_dim * num_heads]
    
    if attn_mask is not None:   # [tgt_len, src_len] or [num_heads * batch_size, tgt_len, src_len]
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)  # [1, tgt_len, src_len]
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
    
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    # [batch_size * num_heads, tgt_len, q_dim]
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [batch_size * num_heads, src_len, k_dim]
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [batch_size * num_heads, src_len, k_dim]
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    # [batch_size * num_heads, tgt_len, q_dim] × [batch_size * num_heads, k_dim, src_len]
    #  = [batch_size * num_heads, tgt_len, src_len]
    
    if attn_mask is not None:
        attn_output_weights += attn_mask    # [batch_size * num_heads, tgt_len, src_len]
    
    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        # 变成 [batch_size, num_heads, tgt_len, src_len] 的形状
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),     # 扩展维度，从[batch_size, src_len]变成[batch_size, 1, 1, src_len]
            float('-inf')
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
        # [batch_size * num_heads, tgt_len, src_len]
        
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)   # [batch_size * num_heads, tgt_len, src_len]
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    # Z = [batch_size * num_heads, tgt_len, src_len] × [batch_size * num_heads, src_len, v_dim]
    # = [batch_size * num_heads, tgt_len, v_dim]
    # 这就是num_heads个Attention(Q,K,V)的结果
    
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    # 先transpose成[tgt_len, batch_size * num_heads, k_dim]
    # 再view成[tgt_len, batch_size, num_heads * k_dim]
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    
    Z = out_proj(attn_output)
    # 多个z线性组合成Z [tgt_len, batch_size, embed_dim]
    
    return Z, attn_output_weights.sum(dim=1) / num_heads

    
class MyMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        """
        param embed_dim: 词嵌入的维度，也就是前面的d_model参数，论文中的默认值为 512
        param num_heads: 多头注意力机制中多头的数量，也就是前面的n_head参数，论文默认值为 8
        param dropout:
        param bias: 最后对多头注意力（组合）输出进行线性变换时，是否使用偏置
        """
        super(MyMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim  # 前面的d_model参数
        self.head_dim = embed_dim // num_heads  # head_dim 指的就是 d_q,d_k,d_v
        self.k_dim = self.head_dim
        self.v_dim = self.head_dim
        
        self.num_heads = num_heads
        self.dropout = dropout
        
        assert self.head_dim * num_heads == self.embed_dim # embed_dim 除以 num_heads 必须为整数
        # 上面的限制条件就是论文中的 d_k = d_v = d_model / n_head 条件
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)   # embed_dim = k_dim * num_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)   # W_k, embed_dim = k_dim * num_heads
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)   # W_v, embed_dim = v_dim * num_heads
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()
    
    def _reset_parameters(self):
        """
        以特定方式来初始化参数
        """
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """前向传播函数
        在论文中，编码时query, key, value 都是同一个输入，解码时 输入的部分也都是同一个输入
        解码和编码交互时 key, value 指的是 memory, query 指的是tgt
        Args:
            param query: [tgt_len, batch_size, embed_dim], tgt_len 表示目标序列的长度
            param key:   [src_len, batch_size, embed_dim], src_len 表示源序列的长度
            param value: [src_len, batch_size, embed_dim], src_len 表示源序列的长度
            param attn_mask: [tgt_len, src_len] or [num_heads * batch_size, tgt_len, src_len]
            param key_padding_mask: [batch_size, src_len], src_len 表示源序列的长度
        """
        return multi_head_attention_forward(query, key, value, self.num_heads,
                                            self.dropout,
                                            out_proj=self.out_proj,
                                            training=self.training,
                                            key_padding_mask=key_padding_mask,
                                            q_proj=self.q_proj,
                                            k_proj=self.k_proj,
                                            v_proj=self.v_proj,
                                            attn_mask=attn_mask)

    
class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1):
        super(MyTransformerEncoderLayer, self).__init__()
        """
        param d_model:          d_k = d_v = d_model / n_head = 64, 模型中词向量嵌入的维度，论文中默认值为 512
        param n_head:           多头注意力中多头的数量，论文默认值为 8
        param dim_feedforward:  全连接中向量的维度，论文默认值为 2048
        param dropout:          丢弃率，论文中的默认值为 0.1
        """    
        self.self_attn = MyMultiheadAttention(d_model, n_head, dropout=dropout)
        
        # Implementation of Feedforward model
        self.dropout1 = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = F.relu
        
        self.dropout2 = nn.Dropout(p=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        param src: 编码部分的输入，形状为 [src_len, batch_size, embed_dim]
        param src_mask: 编码部分输入的padding情况，形状为 [batch_size, src_len]
        """    
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]   # 多头注意力的输出
        # src2.shape = [src_len, batch_size, num_heads * v_dim]
        src = src + self.dropout1(src2) # 残差连接
        src = self.norm1(src)   # [src_len, batch_size, num_heads * k_dim]
        
        src2 = self.activation(self.linear1(src))   # [src_len, batch_size, dim_feedforward]
        src2 = self.linear2(self.dropout2(src2))    # [src_len, batch_size, d_model]
        src = src + self.dropout2(src2)   # 残差连接
        src = self.norm2(src)             # 层归一化
        return src   # [src_len, batch_size, num_heads * k_dim] <=> [src_len, batch_size, embed_dim]


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])        


class MyTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        """
        encoder_layer: 包含多头注意力机制的一个编码层
        num_layers: 克隆得到多个encoder_layer，论文中默认为 6
        norm: 归一化层
        """
        super(MyTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)    # 克隆得到多个 encoder_layers，论文中默认为 6
        self.num_layers= num_layers
        self.norm = norm
    
    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        param src: 编码部分的输入，形状为 [src_len, batch_size, embed_dim]
        param mask: 编码部分的注意力掩盖矩阵输入
        param src_key_padding_mask: 编码部分输入的padding情况，形状为[batch_size, src_len]
        """
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)  # 多个encoder layers层堆叠后的前向传播过程
        if self.norm is not None:
            output = self.norm(output)
        return output   # [src_len, batch_size, num_heads * k_dim] <=> [src_len, batch_size, embed_dim]
    
class MyTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1):
        """
        param d_model:  d_k = d_v = d_model / n_head = 64，模型中词嵌入向量的维度，论文中的默认值为 512
        param n_head:   多头注意力中多头的数量，论文默认值为 8
        param dim_feedforward:  全连接中向量的维度，论文中的默认值为 2048
        param dropout:  丢弃率，论文中的默认值为 0.1
        """
        super(MyTransformerDecoderLayer, self).__init__()
        # 解码部分输入序列之间的多头注意力（也就是论文结构图中的Masked Multi-head Attention）
        self.self_attn = MyMultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout)
        # 编码部分输出（memory）和解码部分之间的多头注意力机制
        self.multihead_attn = MyMultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.activation = F.relu
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        Args:
            param tgt: 解码部分的输入，形状为 [tgt_len, batch_size, embed_dim]
            param memory: 编码部分的输出 memory，形状为 [src_len, batch_size, embed_dim]
            param tgt_mask: 注意力mask输入，用于掩盖当前position之后的信息，[tgt_len, tgt_len]
            param memory_mask: 编码器-解码器交互时的注意力掩码，一般为None
            param tgt_key_padding_mask: 解码部分输入的padding情况，形状为 [batch_size, tgt_len]
            param memory_key_padding_mask: 编码部分输入的padding情况，形状为 [batch_size, src_len]
        """
        # 解码部分输入序列之间的多头注意力（也就是论文结构图中的Masked Multi-head Attention）
        tgt2 = self.self_attn(tgt, tgt, tgt,    # [tgt_len, batch_size, embed_dim]
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        
        tgt = tgt + self.dropout1(tgt2)  # 残差连接
        tgt = self.norm1(tgt)            # 层归一化
        
        # 解码部分的输入经过多头注意力后同编码部分的输出（memory）通过多头注意力机制进行交互
        tgt2 = self.multihead_attn(tgt, memory, memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)  # 残差连接
        tgt = self.norm2(tgt)            # 层归一化
        
        # 最后两层的全连接
        tgt2 = self.activation(self.linear1(tgt))   # [tgt_len, batch_size, dim_feedforward]
        tgt2 = self.linear2(self.dropout(tgt2))     # [tgt_len, batch_size, embed_dim]
        tgt = tgt + self.dropout3(tgt2)  # 残差连接
        tgt = self.norm3(tgt)            # 层归一化
        return tgt  # [tgt_len, batch_size, num_heads * k_dim] <=> [tgt_len, batch_size, embed_dim]
    

class MyTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(MyTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers= num_layers
        self.norm = norm
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        Args:
            param tgt: 解码部分的输入，形状为 [tgt_len, batch_size, embed_dim]
            param memory: 编码部分最后一层的输出 [src_len, batch_size, embed_dim]
            param tgt_mask: 注意力Mask输入，用于掩盖当前position之后的信息，[tgt_len, tgt_len] 
            param memory_mask: 编码器-解码器交互时的注意力掩码，一般为None
            param tgt_key_padding_mask: 解码部分输入的padding情况，形状为 [batch_size, tgt_len]
            param memory_key_padding_mask: 编码部分输入的padding情况，形状为 [batch_size, src_len]
        """
        output = tgt   # [tgt_len, batch_size, embed_dim]
        
        for mod in self.layers:
            output = mod(output, memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        
        return output   # [tgt_len, batch_size, num_heads * k_dim] <=> [tgt_len, batch_size, embed_dim]


class MyTransformer(nn.Module):
    def __init__(self, d_model=512, n_head=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        """
        param d_model: d_k = d_v = d_model / n_head = 64，模型中词向量嵌入的维度，论文中的默认值为 512
        param n_head:               多头注意力机制中多头的数量，论文默认值为 8
        param num_encoder_layers:   encoder堆叠的数量，也就是论文中的N，论文默认值为 6
        param num_decoder_layers:   decoder堆叠的数量，也就是论文中的N，论文默认值为 6
        param dim_feedforward:      全连接中向量的维度，论文默认值为 2048
        param dropout:              丢弃率，论文中的默认值为 0.1
        """
        super(MyTransformer, self).__init__()
        
        # ================== 编码部分 ===================
        encoder_layer = MyTransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = MyTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        # ================== 解码部分 ===================
        decoder_layer = MyTransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = MyTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        self._reset_parameters()
        
        self.d_model = d_model
        self.n_head = n_head
    
    def _reset_parameters(self):
        '''
        Initiate parameters in the Transformer model.
        '''
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        param src:  [src_len, batch_size, embed_dim]
        param tgt:  [tgt_len, batch_size, embed_dim]
        param src_mask: None
        param tgt_mask: [tgt_len, tgt_len]
        param memory_mask: None
        param src_key_padding_mask: [batch_size, src_len]
        param tgt_key_padding_mask: [batch_size, tgt_len]
        param memory_key_padding_mask: [batch_size, src_len]
        return: [tgt_len, batch_size, num_heads * k_dim] <=> [tgt_len, batch_size, embed_dim]
        """
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # memory.shape = [src_len, batch_size, num_heads * k_dim] <=> [src_len, batch_size, embed_dim]
        output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output   # [tgt_len, batch_size, num_heads * k_dim] <=> [tgt_len, batch_size, embed_dim]
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask  # [sz, sz]   
              
    
if __name__ == '__main__':
    src_len = 5
    batch_size = 2
    d_model = 32
    tgt_len = 6
    num_head = 8
    src = torch.rand((src_len, batch_size, d_model))   # shape: [src_len, batch_size, embed_dim]
    src_key_padding_mask = torch.tensor([[True, True, True, False, False],
                                         [True, True, True, True, False]])  # shape: [batch_size, src_len]
    
    tgt = torch.rand((tgt_len, batch_size, d_model))   # shape: [tgt_len, batch_size, embed_dim]
    tgt_key_padding_mask = torch.tensor([[True, True, True, False, False, False],
                                         [True, True, True, True, False, False]])  # shape: [batch_size, tgt_len]

    # =============== 测试 MyMultiheadAttention ================
    my_mh = MyMultiheadAttention(embed_dim=d_model, num_heads=num_head)
    r = my_mh(src, src, src, key_padding_mask=src_key_padding_mask)
    
    #  ============ 测试 MyTransformerEncoderLayer ============
    my_transformer_encoder_layer = MyTransformerEncoderLayer(d_model=d_model, n_head=num_head)
    r = my_transformer_encoder_layer(src=src, src_key_padding_mask=src_key_padding_mask)
    
    #  ============ 测试 MyTransformerDecoder ============
    my_transformer_encoder_layer = MyTransformerEncoderLayer(d_model=d_model, n_head=num_head)
    my_transformer_encoder = MyTransformerEncoder(encoder_layer=my_transformer_encoder_layer,
                                                  num_layers=2,
                                                  norm=nn.LayerNorm(d_model))
    memory = my_transformer_encoder(src=src, mask=None, src_key_padding_mask=src_key_padding_mask)
    print(memory.shape)

    
    my_transformer_decoder_layer = MyTransformerDecoderLayer(d_model=d_model, n_head=num_head)
    my_transformer_decoder = MyTransformerDecoder(decoder_layer=my_transformer_decoder_layer,
                                                  num_layers=1,
                                                  norm=nn.LayerNorm(d_model))
    out = my_transformer_decoder(tgt=tgt, memory=memory, tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=src_key_padding_mask)
    print(out.shape)

    # ============ 测试 MyTransformer ============
    my_transformer = MyTransformer(d_model=d_model, n_head=num_head, num_encoder_layers=6,
                                   num_decoder_layers=6, dim_feedforward=500)
    src_mask = my_transformer.generate_square_subsequent_mask(src_len)
    tgt_mask = my_transformer.generate_square_subsequent_mask(tgt_len)
    out = my_transformer(src=src, tgt=tgt, tgt_mask=tgt_mask,
                         src_key_padding_mask=src_key_padding_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=src_key_padding_mask)
    print(out.shape)

   