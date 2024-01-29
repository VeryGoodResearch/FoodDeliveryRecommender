# -*- coding: UTF-8 -*-

""" FM
Reference:
	Factorization Machines. Steffen Rendle.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from models.BaseModel import MultiContextSeqModel


class FM(MultiContextSeqModel):
	# runner = 'ImpressionRunner'
	runner = 'ExpReRunner'
	extra_log_args=['sample_for_train','train_exploration','test_exploration',
                 'use_context_features','include_item_features']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--use_context_features',type=int,default=64,
                      help='Using context features or not.')
		return MultiContextSeqModel.parse_model_args(parser)	

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.vec_size = args.emb_size
		ctx_pre_mh_features = [corpus.feature_max[f] for f in corpus.context_feature_names if 
									 f.endswith("_c") and f.startswith('c_pre')]
		self.ctx_pre_mh_feature_dim = sum(ctx_pre_mh_features)
		self.ctx_pre_mh_feature_num = len(ctx_pre_mh_features)
		self.ctx_pre_nu_feature_num = len([f for f in corpus.context_feature_names if 
									 f.endswith("_f") and f.startswith('c_pre')])
		item_mh_features = [corpus.feature_max[f] for f in corpus.item_feature_names if f[-2:]=='_c']
		if self.include_id:
			item_mh_features.append(self.item_num)
		self.item_mh_feature_dim = sum(item_mh_features)
		self.item_mh_feature_num = len(item_mh_features)
		self.item_nu_feature_num = len([f for f in corpus.item_feature_names if f[-2:]=='_f'])
		self.use_context = args.include_context_features>0 and args.use_context_features>0
		self._define_params()
		self.apply(self.init_weights)

	def _define_params(self):
		self.user_embeddings = nn.Embedding(self.user_num, self.vec_size)
		self.item_mh_embeddings = nn.Embedding(self.item_mh_feature_dim,self.vec_size)
		if self.item_nu_feature_num>0:
			self.item_nu_embeddings = nn.ModuleList(
				[nn.Linear(1,self.vec_size) for i in range(self.item_nu_feature_num)]
			)

		self.ctx_pre_mh_embeddings = nn.Embedding(self.ctx_pre_mh_feature_dim, self.vec_size)
		if self.ctx_pre_nu_feature_num>0:
			self.ctx_pre_nu_embedding = nn.ModuleList(
	  			[nn.Linear(1, self.vec_size) for i in range(self.ctx_pre_nu_feature_num)])

		self.context_feature_num= 1 + self.item_mh_feature_num + self.item_nu_feature_num \
  					+ self.use_context*(self.ctx_pre_mh_feature_num+self.ctx_pre_nu_feature_num)
		self.linear_embedding = nn.Linear(self.context_feature_num, 1)
		self.overall_bias = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=True)
	
	def forward(self, feed_dict):
		u_ids = feed_dict['user_id']  # [batch_size]
		u_vectors = self.user_embeddings(u_ids).unsqueeze(1).unsqueeze(1)
 
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
		item_emb = item_emb.squeeze(1) # batch, item num, feature num, emb size
		item_num = item_emb.shape[1]
		if self.use_context:
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
			context_features = torch.cat([u_vectors.repeat(1,item_num,1,1),item_emb,situation_vectors.repeat(1,item_num,1,1)],dim=-2)
		else:
			context_features = torch.cat([u_vectors.repeat(1,item_num,1,1),item_emb],dim=-2)

		linear_value = self.overall_bias + self.linear_embedding(context_features.transpose(-1,-2)).squeeze(dim=-1)
		linear_value = linear_value.sum(dim=-1)
		fm_vectors = context_features
		fm_vectors = 0.5 * (fm_vectors.sum(dim=-2).pow(2) - fm_vectors.pow(2).sum(dim=-2))
		predictions = linear_value + fm_vectors.sum(dim=-1)

		return {'prediction':predictions, 'candidate_num':feed_dict['candidates']}
