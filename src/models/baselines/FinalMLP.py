import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from models.BaseModel import MultiContextCFModel
from utils.layers import MLP_Block

class FinalMLP(MultiContextCFModel):
	runner = 'ExpReRunner'
	extra_log_args = ['train_exploration','test_exploration','use_fs']
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--mlp1_hidden_units', type=str,default='[64,64,64]')
		parser.add_argument('--mlp1_hidden_activations',type=str,default='ReLU')
		parser.add_argument('--mlp1_dropout',type=float,default=0)
		parser.add_argument('--mlp1_batch_norm',type=int,default=0)
		parser.add_argument('--mlp2_hidden_units', type=str,default='[64,64,64]')
		parser.add_argument('--mlp2_hidden_activations',type=str,default='ReLU')
		parser.add_argument('--mlp2_dropout',type=float,default=0)
		parser.add_argument('--mlp2_batch_norm',type=int,default=0)
		parser.add_argument('--use_fs',type=int,default=1)
		parser.add_argument('--fs_hidden_units',type=str,default='[64]')
		parser.add_argument('--fs1_context',type=str,default='')
		parser.add_argument('--fs2_context',type=str,default='')
		parser.add_argument('--num_heads',type=int,default=1)
		parser.add_argument('--output_sigmoid',type=int,default=0)
		return MultiContextCFModel.parse_model_args(parser)

	def get_fs_context(self, context_name):
		if context_name == 'none':
			return []
		if context_name == 'situation':
			return self.ctx_pre_mh_feature_names
		if context_name == 'item':
			return self.item_feature_names
		if context_name == 'all':
			return self.ctx_pre_mh_feature_names+self.item_feature_names

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.embedding_dim = args.emb_size

		self.ctx_pre_mh_feature_names = [f for f in corpus.context_feature_names if 
									 (f.endswith("_c")) and f.startswith('c_pre')]
		self.ctx_pre_mh_features = [corpus.feature_max[f] for f in corpus.context_feature_names if 
									 f.endswith("_c") and f.startswith('c_pre')]
		self.ctx_pre_mh_feature_dim = sum(self.ctx_pre_mh_features)
		self.ctx_pre_mh_feature_num = len(self.ctx_pre_mh_features)
		self.ctx_pre_nu_feature_num = len([f for f in corpus.context_feature_names if 
									 f.endswith("_f") and f.startswith('c_pre')])
		self.item_feature_names = [f for f in corpus.item_feature_names if f[-2:]=='_c' or f[-2:]=='_f']
		item_mh_features = [corpus.feature_max[f] for f in corpus.item_feature_names if f[-2:]=='_c']
		if self.include_id:
			item_mh_features.append(self.item_num)
		self.item_mh_feature_dim = sum(item_mh_features)
		self.item_mh_feature_num = len(item_mh_features)
		self.item_nu_feature_num = len([f for f in corpus.item_feature_names if f[-2:]=='_f'])
		self.user_num = corpus.n_users

		self.use_fs = args.use_fs
		self.feature_max = corpus.feature_max
		self.fs1_context = self.get_fs_context(args.fs1_context)
		self.fs2_context = self.get_fs_context(args.fs2_context)
		
		self._define_params(args)
		self.apply(self.init_weights)

	def _define_params(self,args):
		# embeddings
		self.user_embedding = nn.Embedding(self.user_num, self.embedding_dim)
		self.item_mh_embeddings = nn.Embedding(self.item_mh_feature_dim,self.embedding_dim)
		if self.item_nu_feature_num>0:
			self.item_nu_embeddings = nn.ModuleList(
				[nn.Linear(1,self.embedding_dim) for i in range(self.item_nu_feature_num)]
			)
		self.ctx_pre_mh_embeddings = nn.Embedding(self.ctx_pre_mh_feature_dim, self.embedding_dim)
		if self.ctx_pre_nu_feature_num>0:
			self.ctx_pre_nu_embedding = nn.ModuleList(
	  			[nn.Linear(1, self.embedding_dim) for i in range(self.ctx_pre_nu_feature_num)])
		self.item_embsize = (self.item_mh_feature_num + self.item_nu_feature_num)*self.embedding_dim
		self.ctx_embsize = (self.ctx_pre_mh_feature_num+self.ctx_pre_nu_feature_num)*self.embedding_dim
		self.feature_dim = self.item_embsize + self.ctx_embsize + self.embedding_dim

		# MLP 1
		self.mlp1 = MLP_Block(input_dim=self.feature_dim,output_dim=None,hidden_units=eval(args.mlp1_hidden_units),
						hidden_activations=args.mlp1_hidden_activations,dropout_rates=args.mlp1_dropout,
						batch_norm=args.mlp1_batch_norm)
		self.mlp2 = MLP_Block(input_dim=self.feature_dim,output_dim=None,hidden_units=eval(args.mlp2_hidden_units),
						hidden_activations=args.mlp2_hidden_activations,dropout_rates=args.mlp2_dropout,
						batch_norm=args.mlp2_batch_norm)
		if self.use_fs:
			self.fs_module = FeatureSelection({},self.feature_dim,
									 self.embedding_dim, eval(args.fs_hidden_units),
									 self.fs1_context,self.fs2_context,self.feature_max)
		self.fusion_module = InteractionAggregation(eval(args.mlp1_hidden_units)[-1],
									eval(args.mlp2_hidden_units)[-1],output_dim=1,num_heads=args.num_heads)
		self.output_activation = self.get_output_activation(args.output_sigmoid)


	def forward(self, feed_dict):
		"""
		Inputs: [X,y]
		"""
		user_ids = feed_dict['user_id']
		user_emb = self.user_embedding(user_ids).unsqueeze(dim=1).unsqueeze(dim=1) # batch, 1, 1, emb size
		item_mh = self.item_mh_embeddings(feed_dict['item_mh_features'])
		if self.item_nu_feature_num>0:
			items_nu = feed_dict['item_nu_features'].float()
			items_nu_emb = []
			for i, embedding_layer in enumerate(self.item_nu_embeddings):
				items_nu_emb.append(embedding_layer(items_nu[:,:,i].unsqueeze(-1)))
			items_nu_emb = torch.stack(items_nu_emb,dim=-2)
			item_emb = torch.cat([item_mh,items_nu_emb],dim=-2)
		else:
			item_emb = item_mh
		# item_emb = item_emb.squeeze(1).flatten(start_dim=-2) # batch, item num, feature num*emb size
  
		pre_ctx_mh = self.ctx_pre_mh_embeddings(feed_dict['context_pre_mh_features'])
		if self.ctx_pre_nu_feature_num>0:
			pre_ctx_nu = feed_dict['context_pre_nu_features'].float()
			pre_ctx_nu_emb = []
			for i, embedding_layer in enumerate(self.ctx_pre_nu_embedding):
				pre_ctx_nu_emb.append(embedding_layer(pre_ctx_nu[:,:,i].unsqueeze(-1)))
			pre_ctx_nu_emb = torch.stack(pre_ctx_nu_emb,dim=-2)
			situation_vectors = torch.cat([pre_ctx_mh,pre_ctx_nu_emb],dim=-2)
		else:
			situation_vectors = pre_ctx_mh
		# situation_vectors = situation_vectors.squeeze(1).flatten(start_dim=-2) # batch, 1, feature num*emb size
		
		item_num = item_emb.shape[1]
		X = torch.cat([user_emb.repeat(1,item_num,1,1),item_emb,situation_vectors.repeat(1,item_num,1,1)],
					   dim=-2) 
		flat_emb = X.flatten(start_dim=-2)
		if self.use_fs:
			feat1, feat2 = self.fs_module(feed_dict, flat_emb)
		else:
			feat1, feat2 = flat_emb, flat_emb
		emb_dim1, emb_dim2 = feat1.shape[-1], feat2.shape[-1]
		batch_size, item_num = feat1.shape[0], feat1.shape[1]
		mlp1_output = self.mlp1(feat1.view(-1,emb_dim1)).view(batch_size, item_num, -1)
		mlp2_output = self.mlp1(feat1.view(-1,emb_dim2)).view(batch_size, item_num, -1)
		y_pred = self.fusion_module(mlp1_output, mlp2_output)
		# y_pred = self.fusion_module(self.mlp1(feat1), self.mlp2(feat2))
		y_pred = self.output_activation(y_pred)
		return {'prediction':y_pred, 'candidate_num':feed_dict['candidates']}

	def get_output_activation(self, output_sigmoid):
		if output_sigmoid:
			return nn.Sigmoid()
		else:
			return nn.Identity()

	class Dataset(MultiContextCFModel.Dataset):
		def __init__(self, model, corpus, phase):
			super().__init__(model, corpus, phase)
			self.remain_features = list(set(model.fs1_context)|set(model.fs2_context))

		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			for f in self.remain_features:
				if f.startswith('c_'):
					feed_dict[f] = self.data[f][index]
				elif f.startswith('i_'):
					feed_dict[f] = np.array([max(self.corpus.item_features[iid][f],0) for iid in feed_dict['item_id']])
			return feed_dict

class FeatureSelection(nn.Module):
	def __init__(self, feature_map, feature_dim, embedding_dim, fs_hidden_units=[], 
				 fs1_context=[], fs2_context=[],feature_maxn=dict()):
		super(FeatureSelection, self).__init__()
		self.fs1_context = fs1_context
		if len(fs1_context) == 0:
			self.fs1_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
		else:
			'''
			https://github.com/reczoo/FuxiCTR/blob/v2.0.1/fuxictr/pytorch/layers/embeddings/feature_embedding.py
			'''
			self.fs1_ctx_emb = []
			for ctx in fs1_context:
				if ctx.endswith("_c") and ctx in feature_maxn:
					self.fs1_ctx_emb.append(nn.Embedding(feature_maxn[ctx],embedding_dim))
				elif ctx.endswith("_f"):
					self.fs1_ctx_emb.append(nn.Linear(1,embedding_dim))
				else:
					raise ValueError("Undifined context %s"%(ctx))
			self.fs1_ctx_emb = nn.ModuleList(self.fs1_ctx_emb)
			# self.fs1_ctx_emb = nn.Embedding(feature_map, embedding_dim, # 应该是feature_embedding；
			# 									required_feature_columns=fs1_context)
		self.fs2_context = fs2_context
		if len(fs2_context) == 0:
			self.fs2_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
		else:
			self.fs2_ctx_emb = []
			for ctx in fs2_context:
				if ctx.endswith("_c") and ctx in feature_maxn:
					self.fs2_ctx_emb.append(nn.Embedding(feature_maxn[ctx],embedding_dim))
				elif ctx.endswith("_f"):
					self.fs2_ctx_emb.append(nn.Linear(1,embedding_dim))
				else:
					raise ValueError("Undifined context %s"%(ctx))
			self.fs2_ctx_emb = nn.ModuleList(self.fs2_ctx_emb)
			# self.fs2_ctx_emb = nn.Embedding(feature_map, embedding_dim,
			# 									required_feature_columns=fs2_context)
		self.fs1_gate = MLP_Block(input_dim=embedding_dim * max(1, len(fs1_context)),
								  output_dim=feature_dim,
								  hidden_units=fs_hidden_units,
								  hidden_activations="ReLU",
								  output_activation="Sigmoid",
								  batch_norm=False)
		self.fs2_gate = MLP_Block(input_dim=embedding_dim * max(1, len(fs2_context)),
								  output_dim=feature_dim,
								  hidden_units=fs_hidden_units,
								  hidden_activations="ReLU",
								  output_activation="Sigmoid",
								  batch_norm=False)
		
	def forward(self, feed_dict, flat_emb):
		if len(self.fs1_context) == 0:
			fs1_input = self.fs1_ctx_bias.unsqueeze(1).repeat(flat_emb.size(0),flat_emb.size(1), 1)
		else:
			fs1_input = []
			for i,ctx in enumerate(self.fs1_context):
				if ctx.endswith('_c'):
					try:
						ctx_emb = self.fs1_ctx_emb[i](feed_dict[ctx])
					except:
						print(ctx)
						input_str = ''
						while input_str != 'continue':
							try:
								print(eval(input_str))
							except Exception as e:
								print(e)
							input_str = input()
				else:
					ctx_emb = self.fs1_ctx_emb[i](feed_dict[ctx].float().unsqueeze(-1))
				if len(ctx_emb.shape)==2:
					fs1_input.append(ctx_emb.unsqueeze(1).repeat(1,flat_emb.size(1),1))
				else:
					fs1_input.append(ctx_emb)
			fs1_input = torch.cat(fs1_input,dim=-1)
		gt1 = self.fs1_gate(fs1_input) * 2
		feature1 = flat_emb * gt1
		if len(self.fs2_context) == 0:
			fs2_input = self.fs2_ctx_bias.unsqueeze(1).repeat(flat_emb.size(0),flat_emb.size(1), 1)
		else:
			fs2_input = []
			for i,ctx in enumerate(self.fs2_context):
				if ctx.endswith('_c'):
					ctx_emb = self.fs2_ctx_emb[i](feed_dict[ctx])
				else:
					ctx_emb = self.fs2_ctx_emb[i](feed_dict[ctx].float().unsqueeze(-1))
				if len(ctx_emb.shape)==2:
					fs2_input.append(ctx_emb.unsqueeze(1).repeat(1,flat_emb.size(1),1))
				else:
					fs2_input.append(ctx_emb)
			fs2_input = torch.cat(fs2_input,dim=-1)
		gt2 = self.fs2_gate(fs2_input) * 2
		feature2 = flat_emb * gt2
		return feature1, feature2

class InteractionAggregation(nn.Module):
	def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
		super(InteractionAggregation, self).__init__()
		assert x_dim % num_heads == 0 and y_dim % num_heads == 0, \
			"Input dim must be divisible by num_heads!"
		self.num_heads = num_heads
		self.output_dim = output_dim
		self.head_x_dim = x_dim // num_heads
		self.head_y_dim = y_dim // num_heads
		self.w_x = nn.Linear(x_dim, output_dim)
		self.w_y = nn.Linear(y_dim, output_dim)
		self.w_xy = nn.Parameter(torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim, 
											  output_dim))
		nn.init.xavier_normal_(self.w_xy)

	def forward(self, x, y):
		batch_size, item_num = x.shape[0], x.shape[1]
		output = self.w_x(x) + self.w_y(y)
		head_x = x.view(batch_size, item_num, self.num_heads, self.head_x_dim).flatten(start_dim=0,end_dim=1)
		head_y = y.view(batch_size, item_num, self.num_heads, self.head_y_dim).flatten(start_dim=0,end_dim=1)
		xy = torch.matmul(torch.matmul(head_x.unsqueeze(2), 
									   self.w_xy.view(self.num_heads, self.head_x_dim, -1)) \
							   .view(-1, self.num_heads, self.output_dim, self.head_y_dim),
						  head_y.unsqueeze(-1)).squeeze(-1)
		xy_reshape = xy.sum(dim=1).view(batch_size,item_num,-1)
		output += xy_reshape
		return output.squeeze(-1)