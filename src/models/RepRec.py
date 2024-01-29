# A new version of ContextItemAtt, 0 for exploration items
import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
import pandas as pd

from models.BaseModel import MultiContextSeqModel

class RepRec(MultiContextSeqModel):
	runner = 'ExpReRunner'
	extra_log_args = ['history_max','emb_size','sample_for_train']
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--prediction_no_exp', type=int, default=1,
							help='Predict 0 for exploration')
		return MultiContextSeqModel.parse_model_args(parser)	
	
	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.emb_size = args.emb_size
		self.prediction_no_exp = args.prediction_no_exp
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

		self._define_params()
		self.apply(self.init_weights)

	def _define_params(self):
		self.item_mh_embeddings = nn.Embedding(self.item_mh_feature_dim,self.emb_size)
		if self.item_nu_feature_num>0:
			self.item_nu_embeddings = nn.ModuleList(
				[nn.Linear(1,self.emb_size) for i in range(self.item_nu_feature_num)]
			)

		self.ctx_pre_mh_embeddings = nn.Embedding(self.ctx_pre_mh_feature_dim, self.emb_size)
		if self.ctx_pre_nu_feature_num>0:
			self.ctx_pre_nu_embedding = nn.ModuleList(
	  			[nn.Linear(1, self.emb_size) for i in range(self.ctx_pre_nu_feature_num)])
		self.ctx_item_weights = nn.Linear(self.ctx_pre_mh_feature_num+self.ctx_pre_nu_feature_num,
                                    self.item_mh_feature_num+self.item_nu_feature_num,bias=False)
		# initializer = nn.init.xavier_uniform_
		self.item_feature_weights = nn.Parameter(
      			torch.ones(self.item_mh_feature_num+self.item_nu_feature_num,1), 
                    requires_grad=True)#.to(self.device)
 
	def forward(self, feed_dict):
		lengths = feed_dict['lengths']
		his_pre_ctx_mh = self.ctx_pre_mh_embeddings(feed_dict['history_context_pre_mh_features'])
		pre_ctx_mh = self.ctx_pre_mh_embeddings(feed_dict['context_pre_mh_features'])
		mask = torch.arange(his_pre_ctx_mh.shape[1])[None,:].to(self.device) < lengths[:,None] # B * h
	
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
			his_vectors = torch.cat([his_pre_ctx_mh,his_pre_ctx_nu_emb],dim=-2)
			current_vectors = torch.cat([pre_ctx_mh,pre_ctx_nu_emb],dim=-2)
		else:
			his_vectors = his_pre_ctx_mh
			current_vectors = pre_ctx_mh

		similarity_ctx = (his_vectors*current_vectors).sum(dim=-1) # B * h * f num
		weighted_similarity = self.ctx_item_weights(similarity_ctx) # B * h * if num, 此处没有对weight加约束，如果要加softmax等约束需要单独定义

		weighted_similarity=torch.where(mask.unsqueeze(-1).repeat(1,1,weighted_similarity.shape[-1])==1,
                                  weighted_similarity,-torch.tensor(float("Inf")).float().to(self.device))
		weighted_similarity_softmax = (weighted_similarity-weighted_similarity.max()).softmax(dim=1)

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
			item_emb = torch.cat([item_mh,items_nu_emb],dim=-2)
		else:
			history_items = history_items_mh
			item_emb = item_mh
  
		pred_emb = (history_items * weighted_similarity_softmax[:,:,:,None]*mask[:,:,None,None]).sum(dim=1) # weighted average

		prediction = ((pred_emb[:,None,:,:] * item_emb).sum(dim=-1)\
				*self.item_feature_weights.squeeze(-1).softmax(dim=0)[None,None,:]).sum(dim=-1)
		if self.prediction_no_exp:
			item_ids = feed_dict['item_id']
			history_iids = feed_dict['history_items']
			matching = ((history_iids[:,None,:] == item_ids[:,:,None]).sum(dim=-1)>0) # exist in history
			prediction = prediction.sigmoid() * matching
		elif self.phase == 'eval':
			prediction = prediction.sigmoid()
		return {'prediction':prediction, 'candidate_num':feed_dict['candidates']}