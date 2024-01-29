# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as fn
from utils.utils import get_activation

class MultiHeadAttention(nn.Module):
	def __init__(self, d_model, n_heads, kq_same=False, bias=True, attention_d=-1):
		super().__init__()
		"""
		It has projection layer for getting keys, queries and values. Followed by attention.
		"""
		self.d_model = d_model
		self.h = n_heads
		if attention_d<0:
			self.attention_d = self.d_model
		else:
			self.attention_d = attention_d

		self.d_k = self.attention_d // self.h
		self.kq_same = kq_same

		if not kq_same:
			self.q_linear = nn.Linear(d_model, self.attention_d, bias=bias)
		self.k_linear = nn.Linear(d_model, self.attention_d, bias=bias)
		self.v_linear = nn.Linear(d_model, self.attention_d, bias=bias)

	def head_split(self, x):  # get dimensions bs * h * seq_len * d_k
		new_x_shape = x.size()[:-1] + (self.h, self.d_k)
		return x.view(*new_x_shape).transpose(-2, -3)

	def forward(self, q, k, v, mask=None):
		origin_shape = q.size()

		# perform linear operation and split into h heads
		if not self.kq_same:
			q = self.head_split(self.q_linear(q))
		else:
			q = self.head_split(self.k_linear(q))
		k = self.head_split(self.k_linear(k))
		v = self.head_split(self.v_linear(v))

		# calculate attention using function we will define next
		output = self.scaled_dot_product_attention(q, k, v, self.d_k, mask)

		# concatenate heads and put through final linear layer
		output = output.transpose(-2, -3).reshape(origin_shape)
		return output

	@staticmethod
	def scaled_dot_product_attention(q, k, v, d_k, mask=None):
		"""
		This is called by Multi-head attention object to find the values.
		"""
		scores = torch.matmul(q, k.transpose(-2, -1)) / d_k ** 0.5  # bs, head, q_len, k_len
		if mask is not None:
			scores = scores.masked_fill(mask == 0, -np.inf)
		scores = (scores - scores.max()).softmax(dim=-1)
		scores = scores.masked_fill(torch.isnan(scores), 0)
		output = torch.matmul(scores, v)  # bs, head, q_len, d_k
		return output

class AttLayer(nn.Module):
	"""Calculate the attention signal(weight) according the input tensor.
	Reference: RecBole https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/layers.py#L236

	Args:
		infeatures (torch.FloatTensor): An input tensor with shape of[batch_size, XXX, embed_dim] with at least 3 dimensions.

	Returns:
		torch.FloatTensor: Attention weight of input. shape of [batch_size, XXX].
	"""

	def __init__(self, in_dim, att_dim):
		super(AttLayer, self).__init__()
		self.in_dim = in_dim
		self.att_dim = att_dim
		self.w = torch.nn.Linear(in_features=in_dim, out_features=att_dim, bias=False)
		self.h = nn.Parameter(torch.randn(att_dim), requires_grad=True)

	def forward(self, infeatures):
		att_signal = self.w(infeatures)  # [batch_size, XXX, att_dim]
		att_signal = fn.relu(att_signal)  # [batch_size, XXX, att_dim]

		att_signal = torch.mul(att_signal, self.h)  # [batch_size, XXX, att_dim]
		att_signal = torch.sum(att_signal, dim=-1)  # [batch_size, XXX]
		att_signal = fn.softmax(att_signal, dim=-1)  # [batch_size, XXX]

		return att_signal

class TransformerLayer(nn.Module):
	def __init__(self, d_model, d_ff, n_heads, dropout=0, kq_same=False):
		super().__init__()
		"""
		This is a Basic Block of Transformer. It contains one Multi-head attention object. 
		Followed by layer norm and position wise feedforward net and dropout layer.
		"""
		# Multi-Head Attention Block
		self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same=kq_same)

		# Two layer norm layer and two dropout layer
		self.layer_norm1 = nn.LayerNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)

		self.linear1 = nn.Linear(d_model, d_ff)
		self.linear2 = nn.Linear(d_ff, d_model)

		self.layer_norm2 = nn.LayerNorm(d_model)
		self.dropout2 = nn.Dropout(dropout)

	def forward(self, seq, mask=None):
		context = self.masked_attn_head(seq, seq, seq, mask)
		context = self.layer_norm1(self.dropout1(context) + seq)
		output = self.linear1(context).relu()
		output = self.linear2(output)
		output = self.layer_norm2(self.dropout2(output) + context)
		return output

class MLP_Block(nn.Module):
	'''
	FuxiCTR
	https://github.com/reczoo/FuxiCTR/blob/v2.0.1/fuxictr/pytorch/layers/blocks/mlp_block.py
	'''
	def __init__(self, 
				 input_dim, 
				 hidden_units=[], 
				 hidden_activations="ReLU",
				 output_dim=None,
				 output_activation=None, 
				 dropout_rates=0.0,
				 batch_norm=False, 
				 layer_norm=False,
				 norm_before_activation=True,
				 use_bias=True):
		super(MLP_Block, self).__init__()
		dense_layers = []
		if not isinstance(dropout_rates, list):
			dropout_rates = [dropout_rates] * len(hidden_units)
		if not isinstance(hidden_activations, list):
			hidden_activations = [hidden_activations] * len(hidden_units)
		hidden_activations = get_activation(hidden_activations, hidden_units)
		hidden_units = [input_dim] + hidden_units
		for idx in range(len(hidden_units) - 1):
			dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
			if norm_before_activation:
				if batch_norm:
					dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
				elif layer_norm:
					dense_layers.append(nn.LayerNorm(hidden_units[idx + 1]))
			if hidden_activations[idx]:
				dense_layers.append(hidden_activations[idx])
			if not norm_before_activation:
				if batch_norm:
					dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
				elif layer_norm:
					dense_layers.append(nn.LayerNorm(hidden_units[idx + 1]))
			if dropout_rates[idx] > 0:
				dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
		if output_dim is not None:
			dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
		if output_activation is not None:
			dense_layers.append(get_activation(output_activation))
		self.mlp = nn.Sequential(*dense_layers) # * used to unpack list
	
	def forward(self, inputs):
		return self.mlp(inputs)

class MultiHeadTargetAttention(nn.Module):
    '''
    Reference: https://github.com/reczoo/FuxiCTR/blob/v2.0.1/fuxictr/pytorch/layers/attentions/target_attention.py
    '''
    def __init__(self,
                 input_dim=64,
                 attention_dim=64,
                 num_heads=1,
                 dropout_rate=0,
                 use_scale=True,
                 use_qkvo=True):
        super(MultiHeadTargetAttention, self).__init__()
        if not use_qkvo:
            attention_dim = input_dim
        assert attention_dim % num_heads == 0, \
               "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.use_qkvo = use_qkvo
        if use_qkvo:
            self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
            self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
            self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
            self.W_o = nn.Linear(attention_dim, input_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention(dropout_rate)

    def forward(self, target_item, history_sequence, mask=None):
        """
        target_item: b x emd
        history_sequence: b x len x emb
        mask: mask of history_sequence, 0 for masked positions
        """
        # linear projection
        if self.use_qkvo:
            query = self.W_q(target_item)
            key = self.W_k(history_sequence)
            value = self.W_v(history_sequence)
        else:
            query, key, value = target_item, history_sequence, history_sequence

        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        if mask is not None:
            mask = mask.view(batch_size, 1, 1, -1).expand(-1, self.num_heads, -1, -1)

        # scaled dot product attention
        output, _ = self.dot_attention(query, key, value, scale=self.scale, mask=mask)
        # concat heads
        output = output.transpose(1, 2).contiguous().view(-1, self.num_heads * self.head_dim)
        if self.use_qkvo:
            output = self.W_o(output)
        return output


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention 
        Ref: https://zhuanlan.zhihu.com/p/47812375
    """
    def __init__(self, dropout_rate=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, Q, K, V, scale=None, mask=None):
        # mask: 0 for masked positions
        scores = torch.matmul(Q, K.transpose(-1, -2))
        if scale:
            scores = scores / scale
        if mask is not None:
            mask = mask.view_as(scores)
            scores = scores.masked_fill_(mask.float() == 0, -1.e9) # fill -inf if mask=0
        attention = scores.softmax(dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.matmul(attention, V)
        return output, attention

class CrossAtt(torch.nn.Module):
	"""
	Cross attention
	"""
	def __init__(self, input_qsize, input_vsize, query_size=-1, key_size=-1, value_size=-1):
		super().__init__()
		self.input_qsize = input_qsize
		self.input_vsize = input_vsize
		self.query_size = query_size if query_size != 0 else self.input_qsize
		self.key_size = key_size if key_size != 0 else self.input_qsize
		self.value_size = value_size if value_size != 0 else self.input_vsize
		assert self.query_size == self.key_size \
				or (self.query_size < 0 and self.key_size == self.input_size) \
				or (self.key_size < 0 and self.query_size == self.input_size)  # Query和Key的向量维度需要匹配
		self.att_size = input_qsize
		if self.query_size > 0:
			self.att_size = self.query_size
		if self.key_size > 0:
			self.att_size = self.key_size

		self.init_modules()

	def init_modules(self):
		if self.query_size >= 0:
			self.query_layer = torch.nn.Linear(self.input_qsize, self.att_size, bias=False)
		else:
			self.query_layer = None
		if self.key_size >= 0:
			self.key_layer = torch.nn.Linear(self.input_vsize, self.att_size, bias=False)
		else:
			self.key_layer = None
		if self.value_size >= 0:
			self.value_layer = torch.nn.Linear(self.input_vsize, self.value_size, bias=False)
		else:
			self.value_layer = None

	def forward(self, query, x, valid=None, scale=None, act_v=None):

		def transfer_if_valid_layer(layer,input):
			result = input
			if layer is not None:
				result = layer(input)
				if act_v is not None:
					result = act_v(result)
			return result
		att_query = transfer_if_valid_layer(self.query_layer,query)  # ? * L * a
		att_key = transfer_if_valid_layer(self.key_layer,x)  # ? * L * a
		att_value = transfer_if_valid_layer(self.value_layer,x)  # ? * L * v
		return MultiQueryAtt()(q=att_query, k=att_key, v=att_value, scale=scale, valid=valid)

class MultiQueryAtt(torch.nn.Module):
	def forward(self, q, k, v, valid=None, scale=None):
		"""
		multiple-query attention.
		According to the pairwise matching degree of q and k
		weighted average of v corresponding to k	
		"""
		att_v = torch.matmul(q, k.transpose(-1, -2))  # ? * L_q * L_k
		if scale is not None:
			att_v = att_v * scale  # ? * L_q * L_k
		att_v = att_v - att_v.max(dim=-1, keepdim=True)[0]  # ? * L_q * L_k
		if valid is not None:
			att_v = att_v.masked_fill(valid.le(0), -np.inf)  # ? * L_q * L_k
		att_w = att_v.softmax(dim=-1)  # ? * L_q * L_k
		att_w = att_w.masked_fill(torch.isnan(att_w), 0)  # ? * L_q * L_k
		result = torch.matmul(att_w, v)  # ? * L_q * V
		return result, att_w
