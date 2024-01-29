import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from models.BaseModel import MultiContextSeqModel
from utils import layers


class SituExp(MultiContextSeqModel):
	runner = 'ExpReRunner'
	reader = 'ContextNeighborReader'
	extra_log_args=['situation_layers','neighbor_num','train_exploration']
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--train_ignore_repeat',type=int,default=0)
		parser.add_argument('--neighbor_num',type=int,default=5)
		parser.add_argument('--situation_layers',type=int,default=1)
		parser.add_argument('--prediction_no_repeat',type=int,default=1)
		return MultiContextSeqModel.parse_model_args(parser)
  
	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.prediction_no_repeat = args.prediction_no_repeat
		self.train_ignore_repeat = args.train_ignore_repeat
		self.neighbor_num = args.neighbor_num
		self.vec_size = args.emb_size
		self.situation_layers = args.situation_layers
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

		self.user_num = corpus.n_users
		self.max_his = args.history_max
		self._define_params()
		self.apply(self.init_weights)

	def _define_params(self):
		self.user_embedding = nn.Embedding(self.user_num, self.vec_size)
		self.item_mh_embeddings = nn.Embedding(self.item_mh_feature_dim,self.vec_size)
		if self.item_nu_feature_num>0:
			self.item_nu_embeddings = nn.ModuleList(
				[nn.Linear(1,self.vec_size) for i in range(self.item_nu_feature_num)]
			)

		self.ctx_pre_mh_embeddings = nn.Embedding(self.ctx_pre_mh_feature_dim, self.vec_size)
		if self.ctx_pre_nu_feature_num>0:
			self.ctx_pre_nu_embedding = nn.ModuleList(
	  			[nn.Linear(1, self.vec_size) for i in range(self.ctx_pre_nu_feature_num)])

		self.item_embsize = (self.item_mh_feature_num + self.item_nu_feature_num)*self.vec_size
		self.ctx_embsize = (self.ctx_pre_mh_feature_num+self.ctx_pre_nu_feature_num)*self.vec_size
		self.ctx2item = nn.Linear(self.ctx_embsize,self.item_embsize)
		self.ctx2user = nn.Linear(self.ctx_embsize,self.vec_size)
		
		self.hidden_size = self.item_embsize
		self.rnn = nn.GRU(input_size=self.ctx_embsize+self.item_embsize,hidden_size=self.hidden_size,batch_first=True)

		self.activations = nn.ModuleList([
			nn.ELU(),
			nn.Hardsigmoid(),
			nn.Identity(),
			nn.ReLU(),
			nn.SELU(),
			nn.Sigmoid(),
			nn.Softplus(),
			nn.Softsign(),
			nn.Hardswish(),
			nn.Tanh()
		])

		self.conditional_act = nn.ModuleList()
		self.conditional_weights = nn.ModuleList()
		for layer in range(self.situation_layers):
			la_weights = nn.Linear(self.ctx_embsize, len(self.activations))
			input_weights = nn.Linear(self.vec_size, self.vec_size)
			self.conditional_act.append(la_weights)
			self.conditional_weights.append(input_weights)

		self.u2item = nn.Linear(self.vec_size,self.item_embsize)

		self.trigger_weights = nn.Linear(self.item_embsize*2, 4)

	def _get_user_situation(self, users, situations):
		situ_features = situations.squeeze(dim=1).flatten(start_dim=-2)
		user_situation = users
		for input_weights, la_weights in zip(self.conditional_weights,self.conditional_act):
			situ2user = la_weights(situ_features)
			user_situation = input_weights(user_situation)
			user_situation_new = []
			for i,f in enumerate(self.activations):
				if len(user_situation.shape)==2:
					user_situation_new.append(situ2user[:,i][:,None]*f(user_situation))
				else:
					user_situation_new.append(situ2user[:,i][:,None,None]*f(user_situation))
			user_situation = torch.stack(user_situation_new,dim=-1).sum(dim=-1)
		return user_situation.unsqueeze(dim=1)

	def get_trigger_weights(self, situations, users, neighbors, items):
		K = torch.cat([users,situations, neighbors],dim=1).unsqueeze(dim=1) # batch *1 * 3 * emb
		Q = items.unsqueeze(dim=2) # batch * item * 1 * emb
		weights = ((Q*K).sum(dim=-1) / Q.shape[-1]**0.5).softmax(dim=-1) # batch * item * 3
		return weights

	def forward(self, feed_dict):
		# item embeddings
		item_mh = self.item_mh_embeddings(feed_dict['item_mh_features'])
		history_items_mh = self.item_mh_embeddings(feed_dict['history_item_mh_features'])
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
		item_emb = item_emb.squeeze(1).flatten(start_dim=-2) # batch, item num, feature num*emb size
  
		# context embeddings
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
			his_situation_vectors = torch.cat([his_pre_ctx_mh,his_pre_ctx_nu_emb],dim=-2)
			situation_vectors = torch.cat([pre_ctx_mh,pre_ctx_nu_emb],dim=-2)
		else:
			his_situation_vectors = his_pre_ctx_mh
			situation_vectors = pre_ctx_mh

		# history info
		history_items = history_items.squeeze(1).flatten(start_dim=-2) # batch, length, feature num, emb size
		his_situation_vectors = (his_situation_vectors.flatten(start_dim=-2)) # B * hislen * (situ f num * emb)
		# # RNN
		his_vectors = torch.cat([history_items,his_situation_vectors],dim=-1)
		sort_his_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
		sort_his_vectors = his_vectors.index_select(dim=0, index=sort_idx)
		history_packed = torch.nn.utils.rnn.pack_padded_sequence(
			sort_his_vectors, sort_his_lengths.cpu(), batch_first=True)
		output, hidden = self.rnn(history_packed, None) # GRU
		# Unsort
		unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
		rnn_vector = hidden[-1].index_select(dim=0, index=unsort_idx)
		# his2item = self.his2item_emb(rnn_vector).unsqueeze(dim=1)
		his2item = (rnn_vector).unsqueeze(dim=1)

		# situ info
		situ2item = self.ctx2item(situation_vectors.flatten(start_dim=-2))
		
		# user info
		user_ids = feed_dict['user_id']
		user_emb = self.user_embedding(user_ids)
		user2item = self.u2item(self._get_user_situation(user_emb,situation_vectors))

		# neighbor user info
		neighbor_users = feed_dict['neighbor_users'] # get neighbor users, batch * n users
		neighbor_user_emb = self.user_embedding(neighbor_users) 
		neighbor_user2item = self.u2item(self._get_user_situation(neighbor_user_emb,situation_vectors)) # batch * 1 * n users * emb
		neighbor_sim = feed_dict['neighbor_sim'].softmax(dim=-1) # batch * n users
		neighbor_user2item = (neighbor_user2item * neighbor_sim[:,None,:,None]).sum(dim=2) # batch * 1 * emb
  
		weights = self.trigger_weights(torch.cat([situ2item,user2item],dim=-1).squeeze(1)).softmax(dim=1) # batch * 3
		user_all = (torch.cat([situ2item,user2item,neighbor_user2item,his2item],dim=1)*weights[:,:,None]).sum(dim=1,keepdim=True)
		predictions = (item_emb * user_all).sum(dim=-1) 

		if (self.phase == 'eval' or self.train_ignore_repeat) and self.prediction_no_repeat:
			item_ids = feed_dict['item_id']
			history_iids = feed_dict['history_items']
			matching = ((history_iids[:,None,:] == item_ids[:,:,None]).sum(dim=-1)>0) # exist in history
			predictions = predictions.sigmoid() * (~matching) # only for exploration
		elif self.phase == 'eval':
			predictions = predictions.sigmoid()
		else:
			predictions = predictions

		return {'prediction':predictions, 'candidate_num':feed_dict['candidates']}

	class Dataset(MultiContextSeqModel.Dataset):
		def __init__(self,model,corpus,phase): 
			super().__init__(model, corpus, phase)
			self.neighbor_dict = corpus.neighbor_dict
			self.neighbor_num = model.neighbor_num
		
		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			feed_dict['neighbor_users'] = np.array(self.neighbor_dict[feed_dict['user_id']][0][:self.neighbor_num])
			feed_dict['neighbor_sim'] = np.array(self.neighbor_dict[feed_dict['user_id']][1][:self.neighbor_num])
			return feed_dict
