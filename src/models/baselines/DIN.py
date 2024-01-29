import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
import pandas as pd

from models.BaseModel import MultiContextSeqModel

class DIN(MultiContextSeqModel):
	runner = 'ExpReRunner'
	extra_log_args = ['loss_n','att_layers','dnn_layers','history_max','sample_for_train','use_context_features']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--att_layers', type=str, default='[64]',
							help="Size of each layer.")
		parser.add_argument('--dnn_layers', type=str, default='[64]',
							help="Size of each layer.")
		parser.add_argument('--softmax_stag',type=int,default=0)
		parser.add_argument('--use_context_features',type=int,default=64,
                      help='Using context features or not.')
		return MultiContextSeqModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.softmax_stag = args.softmax_stag
		self.include_id = 1
		self.emb_size = args.emb_size

		self.use_context = args.include_context_features>0 and args.use_context_features>0
		if self.use_context:
			ctx_pre_mh_features = [corpus.feature_max[f] for f in corpus.context_feature_names if 
										f.endswith("_c") and f.startswith('c_pre')]
			self.ctx_pre_mh_feature_dim = sum(ctx_pre_mh_features)
			self.ctx_pre_mh_feature_num = len(ctx_pre_mh_features)
			self.ctx_pre_nu_feature_num = len([f for f in corpus.context_feature_names if 
										f.endswith("_f") and f.startswith('c_pre')])
			self.ctx_feature_num = self.ctx_pre_mh_feature_num+self.ctx_pre_nu_feature_num		
		else:
			self.ctx_feature_num = 0
	
		item_mh_features = [corpus.feature_max[f] for f in corpus.item_feature_names if f[-2:]=='_c']
		if self.include_id:
			item_mh_features.append(self.item_num)
		self.item_mh_feature_dim = sum(item_mh_features)
		self.item_mh_feature_num = len(item_mh_features)
		self.item_nu_feature_num = len([f for f in corpus.item_feature_names if f[-2:]=='_f'])
		self.item_feature_num = self.item_mh_feature_num+self.item_nu_feature_num		

		user_mh_features = [corpus.feature_max[f] for f in corpus.user_feature_names if f[-2:]=='_c']
		if self.include_id:
			user_mh_features.append(self.user_num)
		self.user_mh_feature_dim = sum(user_mh_features)
		self.user_mh_feature_num = len(user_mh_features)
		self.user_nu_feature_num = len([f for f in corpus.user_feature_names if f[-2:]=='_f'])
		self.user_feature_num = self.user_mh_feature_num+self.user_nu_feature_num		

		self.vec_size = args.emb_size
		self.att_layers = eval(args.att_layers)
		self.dnn_layers = eval(args.dnn_layers)

		self._define_params()
		self.apply(self.init_weights)
	
	def _define_params(self):
		self.item_mh_embeddings = nn.Embedding(self.item_mh_feature_dim,self.emb_size)
		if self.item_nu_feature_num>0:
			self.item_nu_embeddings = nn.ModuleList(
				[nn.Linear(1,self.emb_size) for i in range(self.item_nu_feature_num)]
			)
		self.user_mh_embeddings = nn.Embedding(self.user_mh_feature_dim,self.emb_size)
		if self.user_nu_feature_num>0:
			self.user_nu_embeddings = nn.ModuleList(
				[nn.Linear(1,self.emb_size) for i in range(self.user_nu_feature_num)]
			)

		if self.use_context:
			self.ctx_pre_mh_embeddings = nn.Embedding(self.ctx_pre_mh_feature_dim, self.emb_size)
			if self.ctx_pre_nu_feature_num>0:
				self.ctx_pre_nu_embedding = nn.ModuleList(
					[nn.Linear(1, self.emb_size) for i in range(self.ctx_pre_nu_feature_num)])
		
		self.att_mlp_layers = torch.nn.ModuleList()
		pre_size = 4 * (self.item_feature_num+self.ctx_feature_num) * self.vec_size 
		for size in self.att_layers:
			self.att_mlp_layers.append(torch.nn.Linear(pre_size, size))
			self.att_mlp_layers.append(torch.nn.Sigmoid())
			self.att_mlp_layers.append(torch.nn.Dropout(self.dropout))
			pre_size = size
		self.dense = nn.Linear(pre_size, 1)

		self.dnn_mlp_layers = torch.nn.ModuleList()
		pre_size = 3 * (self.item_feature_num+self.ctx_feature_num) * self.vec_size + self.user_feature_num * self.vec_size
		for size in self.dnn_layers:
			self.dnn_mlp_layers.append(torch.nn.Linear(pre_size, size))
			self.dnn_mlp_layers.append(torch.nn.BatchNorm1d(num_features=size))
			self.dnn_mlp_layers.append(Dice(size))
			self.dnn_mlp_layers.append(torch.nn.Dropout(self.dropout))
			pre_size = size
		self.dnn_mlp_layers.append(torch.nn.Linear(pre_size, 1))

	def attention(self, queries, keys, keys_length, softmax_stag=False, return_seq_weight=False):
		'''Reference:
			RecBole layers: SequenceAttLayer
			queries: batch * (if*vecsize)
		'''
		embedding_size = queries.shape[-1]  # H
		hist_len = keys.shape[1]  # T
		queries = queries.repeat(1, hist_len)
		queries = queries.view(-1, hist_len, embedding_size)
		# MLP Layer
		input_tensor = torch.cat(
			[queries, keys, queries - keys, queries * keys], dim=-1
		)
		output = input_tensor
		for layer in self.att_mlp_layers:
			output = layer(output)
		output = torch.transpose(self.dense(output), -1, -2)
		# get mask
		output = output.squeeze(1)
		mask = self.mask_mat.repeat(output.size(0), 1)
		mask = mask >= keys_length.unsqueeze(1)
		# mask
		if softmax_stag:
			mask_value = -np.inf
		else:
			mask_value = 0.0
		output = output.masked_fill(mask=mask, value=torch.tensor(mask_value))
		output = output.unsqueeze(1)
		output = output / (embedding_size**0.5)
		# get the weight of each user's history list about the target item
		if softmax_stag:
			output = fn.softmax(output, dim=2)  # [B, 1, T]
		if not return_seq_weight:
			output = torch.matmul(output, keys)  # [B, 1, H]
		torch.cuda.empty_cache()
		return output.squeeze()

	def attention_and_dnn(self, item_feats_emb, history_feats_emb, hislens, user_feats_emb):
		batch_size, item_num, feats_emb = item_feats_emb.shape
		_, max_len, his_emb = history_feats_emb.shape

		item_feats_emb2d = item_feats_emb.view(-1, feats_emb) # 每个sample的item在一块
		history_feats_emb2d = history_feats_emb.unsqueeze(1).repeat(1,item_num,1,1).view(-1,max_len,his_emb)
		hislens2d = hislens.unsqueeze(1).repeat(1,item_num).view(-1)
		user_feats_emb2d = user_feats_emb.repeat(1,item_num,1).view(-1, user_feats_emb.shape[-1])
		user_his_emb = self.attention(item_feats_emb2d, history_feats_emb2d, hislens2d,softmax_stag=self.softmax_stag)
		din = torch.cat([user_his_emb, item_feats_emb2d, user_his_emb*item_feats_emb2d, user_feats_emb2d], dim=-1)
		# din = torch.cat([user_his_emb, item_feats_emb2d, user_feats_emb2d,situ_feats_emb2d], dim=-1)
		for layer in self.dnn_mlp_layers:
			din = layer(din)
		predictions = din
		return predictions.view(batch_size, item_num)

	def forward(self, feed_dict):
		hislens = feed_dict['lengths'] # B
		# context embedding
		if self.use_context:
			his_pre_ctx_mh = self.ctx_pre_mh_embeddings(feed_dict['history_context_pre_mh_features'])
			pre_ctx_mh = self.ctx_pre_mh_embeddings(feed_dict['context_pre_mh_features'])
			if self.ctx_pre_nu_feature_num>0:
				his_pre_ctx_nu = feed_dict['history_context_pre_nu_features'].float()
				pre_ctx_nu = feed_dict['context_pre_nu_features'].float()
				his_pre_ctx_nu_emb = []
				pre_ctx_nu_emb = []
				for i, embedding_layer in enumerate(self.ctx_pre_nu_embedding):
					his_pre_ctx_nu_emb.append(embedding_layer(his_pre_ctx_nu[:,:,i].unsqueeze(-1)))
					pre_ctx_nu_emb.append(embedding_layer(pre_ctx_nu[:,:,i].unsqueeze(-1)))
				his_pre_ctx_nu_emb = torch.stack(his_pre_ctx_nu_emb,dim=-2)
				pre_ctx_nu_emb = torch.stack(pre_ctx_nu_emb,dim=-2)
				his_ctx_emb = torch.cat([his_pre_ctx_mh,his_pre_ctx_nu_emb],dim=-2)
				ctx_feats_emb = torch.cat([pre_ctx_mh,pre_ctx_nu_emb],dim=-2)
			else:
				his_ctx_emb = his_pre_ctx_mh
				ctx_feats_emb = pre_ctx_mh

		# item embedding
		history_items_mh = self.item_mh_embeddings(feed_dict['history_item_mh_features'])
		item_mh = self.item_mh_embeddings(feed_dict['item_mh_features'])
		if self.item_nu_feature_num>0:
			history_items_nu = feed_dict['history_item_nu_features'].float()
			items_nu = feed_dict['item_nu_features'].float()
			history_items_nu_emb, items_nu_emb = [],[]
			for i, embedding_layer in enumerate(self.item_nu_embeddings):
				history_items_nu_emb.append(embedding_layer(history_items_nu[:,:,i].unsqueeze(-1)))
				items_nu_emb.append(embedding_layer(items_nu[:,:,i].unsqueeze(-1)))
			history_items_nu_emb = torch.stack(history_items_nu_emb,dim=-2)
			items_nu_emb = torch.stack(items_nu_emb,dim=-2)
			history_items = torch.cat([history_items_mh,history_items_nu_emb],dim=-2) # B * h * if num * emb
			item_feats_emb = torch.cat([item_mh,items_nu_emb],dim=-2)
		else:
			history_items = history_items_mh
			item_feats_emb = item_mh

		# user embedding
		user_mh = self.user_mh_embeddings(feed_dict['user_mh_features'])
		if self.user_nu_feature_num>0:
			users_nu = feed_dict['user_nu_features'].float()
			users_nu_emb = []
			for i, embedding_layer in enumerate(self.user_nu_embeddings):
				users_nu_emb.append(embedding_layer(users_nu[:,:,i].unsqueeze(-1)))
			users_nu_emb = torch.stack(users_nu_emb,dim=-2)
			user_feats_emb = torch.cat([user_mh,users_nu_emb],dim=-2).flatten(start_dim=-2)
		else:
			user_feats_emb = user_mh.flatten(start_dim=-2)

		if self.use_context:
			history_feats_emb = torch.cat([history_items,his_ctx_emb],dim=-2).flatten(start_dim=-2)
			current_feats_emb = torch.cat([item_feats_emb,
				ctx_feats_emb.repeat(1,item_feats_emb.shape[1],1,1)],dim=-2).flatten(start_dim=-2)
		else:
			history_feats_emb = history_items.flatten(start_dim=-2)
			current_feats_emb = item_feats_emb.flatten(start_dim=-2)

		self.mask_mat = (torch.arange(history_feats_emb.shape[1]).view(1,-1)).to(self.device)
		predictions = self.attention_and_dnn(current_feats_emb, history_feats_emb, hislens, user_feats_emb)

		return {'prediction':predictions, 'candidate_num':feed_dict['candidates']}
    
class Dice(nn.Module):
	"""The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.

	Input shape:
		- 2 dims: [batch_size, embedding_size(features)]
		- 3 dims: [batch_size, num_features, embedding_size(features)]

	Output shape:
		- Same shape as input.

	References
		- [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
		- https://github.com/zhougr1993/DeepInterestNetwork, https://github.com/fanoping/DIN-pytorch
	"""

	def __init__(self, emb_size, dim=2, epsilon=1e-8, device='cpu'):
		super(Dice, self).__init__()
		assert dim == 2 or dim == 3

		self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
		self.sigmoid = nn.Sigmoid()
		self.dim = dim

		# wrap alpha in nn.Parameter to make it trainable
		if self.dim == 2:
			self.alpha = nn.Parameter(torch.zeros((emb_size,)).to(device))
		else:
			self.alpha = nn.Parameter(torch.zeros((emb_size, 1)).to(device))

	def forward(self, x):
		assert x.dim() == self.dim
		if self.dim == 2:
			x_p = self.sigmoid(self.bn(x))
			out = self.alpha * (1 - x_p) * x + x_p * x
		else:
			x = torch.transpose(x, 1, 2)
			x_p = self.sigmoid(self.bn(x))
			out = self.alpha * (1 - x_p) * x + x_p * x
			out = torch.transpose(out, 1, 2)
		return out
