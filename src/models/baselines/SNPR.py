import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from models.BaseModel import MultiContextSeqModel
from utils.layers import TransformerLayer

class SNPR(MultiContextSeqModel):
	runner = 'ExpReRunner'
	reader = 'ContextSessionReader'
	extra_log_args = ['train_exploration']
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--num_heads',type=int,default=1)
		parser.add_argument('--num_layers',type=int,default=1)
		parser.add_argument('--lambda_r',type=float,default=0.5)
		return MultiContextSeqModel.parse_model_args(parser)


	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.embedding_dim = args.emb_size
		self.num_heads = args.num_heads
		self.num_layers = args.num_layers
		self.lambda_r = args.lambda_r
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
		self.item_embsize = (self.item_mh_feature_num + self.item_nu_feature_num)*self.embedding_dim

		# transfomer_emb = self.feature_dim
		transfomer_emb = self.item_embsize
		self.history_transfomer = nn.ModuleList([
			TransformerLayer(d_model=transfomer_emb, d_ff=transfomer_emb,
					n_heads=self.num_heads)
			for _ in range(self.num_layers)
		])

		self.recent_transfomer = nn.ModuleList([
			TransformerLayer(d_model=transfomer_emb, d_ff=transfomer_emb,
					n_heads=self.num_heads)
			for _ in range(self.num_layers)
		])

		self.e_linear = nn.Linear(transfomer_emb, transfomer_emb)
		self.e0_linear = nn.Linear(transfomer_emb, transfomer_emb)

		self.tg_linear = nn.Linear(transfomer_emb*2, transfomer_emb)
		self.att_long_linear = nn.Linear(transfomer_emb*2, transfomer_emb)
		self.att_long_q = nn.Linear(transfomer_emb, 1)
		self.att_short_linear = nn.Linear(transfomer_emb*2, transfomer_emb)
		self.att_short_q = nn.Linear(transfomer_emb, 1)
		self.final_r_linear = nn.Linear(transfomer_emb*4, transfomer_emb)
		self.final_e_linear = nn.Linear(transfomer_emb, transfomer_emb)

	def get_embeddings(self, feed_dict):
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
		if len(item_emb.shape)==2:
			item_emb = item_emb.unsqueeze(dim=1)
  
		return history_items, item_emb

	def forward(self, feed_dict):
		history_items, cur_items = self.get_embeddings(feed_dict)
		all_history = history_items.flatten(start_dim=-2) # B * len * emb
		short_history_mask = torch.arange(history_items.shape[1])[None,:].to(self.device) < feed_dict['short_lengths'][:,None]
		short_history = all_history*short_history_mask.unsqueeze(-1) # B * len * emb
		long_history_mask = torch.arange(history_items.shape[1])[None,:].to(self.device) < feed_dict['lengths'][:,None] # B * h
		long_history = all_history*long_history_mask.unsqueeze(-1)
		long_short_history_mask = long_history_mask * (~short_history_mask)

		# all_history = torch.cat(history_items, history_situations) # B * len * emb
  
		short_seq = short_history
		batch_size, seq_len = short_history_mask.shape
		for block in self.recent_transfomer:
			short_seq = block(short_seq, short_history_mask.view(batch_size, 1, 1, seq_len))
		short_seq = short_seq * short_history_mask[:,:,None].float()

		long_seq = long_history
		for block in self.history_transfomer:
			long_seq = block(long_seq, long_history_mask.view(batch_size, 1, 1, seq_len))
		long_seq = long_seq * long_short_history_mask[:,:,None].float()
   
		# unexpectedness
		unexpected_e = self.unexpectedness_model(long_seq, feed_dict['history_session_id'])

		# relevance
		r_gen = all_history.mean(dim=1)
		z_t_1 = short_seq[:,0,:]
		# time geo-weighted
		history_time_sim = feed_dict['history_time_sim']
		history_geo_sim = feed_dict['history_geo_sim']
		r_t = self.subsequence_mean(long_seq, feed_dict['history_session_id'], history_time_sim)
		r_geo = self.subsequence_mean(long_seq, feed_dict['history_session_id'],history_geo_sim)
		r = F.gelu(self.tg_linear(torch.cat([r_t,r_geo],dim=-1)))
		short_history_time_sim, short_history_geo_sim = history_time_sim * short_history_mask, history_geo_sim*short_history_mask
		r_cur_t = self.subsequence_mean(short_seq, feed_dict['history_session_id'], short_history_time_sim)
		r_cur_geo = self.subsequence_mean(short_seq, feed_dict['history_session_id'],short_history_geo_sim)
		short_sid = feed_dict['history_session_id'].max(dim=1)[0].long()-1
		r_cur_t = r_cur_t[torch.arange(len(r)), short_sid]
		r_cur_geo = r_cur_geo[torch.arange(len(r)), short_sid]
		r_cur = F.gelu(self.tg_linear(torch.cat([r_cur_t,r_cur_geo],dim=-1))).unsqueeze(dim=1).repeat(1,r.shape[1],1)

		alpha_weights = self.att_long_q(F.tanh(self.att_long_linear(torch.cat([r, r_cur],dim=-1))))
		r_long = (r * alpha_weights.softmax(dim=1)).sum(dim=1)

		try:
			short_weights = self.att_short_q(F.tanh(self.att_short_linear(torch.cat([short_seq.unsqueeze(dim=2).repeat(1,1,cur_items.shape[1],1), 
                            cur_items.unsqueeze(dim=1).repeat(1,short_seq.shape[1],1,1)],dim=-1))))
		except:
			input_str = ''
			while input_str != 'continue':
				try:
					print(eval(input_str))
				except Exception as e:
					print(e)
				input_str = input()
		r_short = (short_seq[:,:,None,:] * short_weights.softmax(dim=1)).sum(dim=1)

		r_0 = torch.cat([r_gen, z_t_1, r_long],dim=-1)
		r_0 = torch.cat([r_0.unsqueeze(dim=1).repeat(1,r_short.shape[1],1), r_short],dim=-1)

		final_r = r_0
		final_e = unexpected_e

		serendipity = self.lambda_r * self.final_r_linear(final_r)+(1-self.lambda_r)*self.final_e_linear(final_e)[:,None,:]

		prediction = (serendipity * cur_items).sum(dim=-1)

		return {'prediction':prediction, 'candidate_num':feed_dict['candidates']}

	def unexpectedness_model(self, sequence, group):
		weights = torch.ones_like(group)
		weights = torch.where(group==group.max(dim=1)[0].unsqueeze(1),0,weights)
		avg_e = self.subsequence_mean(sequence, group, weights)
		e_i = F.gelu(self.e_linear(avg_e))

		e_0 = F.gelu(self.e0_linear(e_i.sum(dim=1) / e_i.shape[1]))
		return e_0

	def subsequence_mean(self, sequence, group, weights=None):
		max_group = group.max()
		seq_mean = []
		for idx in range(1,max_group+1):
			idx_sum = (sequence * (group==idx).unsqueeze(-1)).sum(dim=1)
			idx_num = torch.clamp(((group==idx)*weights).sum(dim=1),min=1.0).float()
			seq_mean.append(idx_sum / idx_num.unsqueeze(1))
		return torch.stack(seq_mean,dim=1)

	class Dataset(MultiContextSeqModel.Dataset):
		def __init__(self,model,corpus,phase): 
			super().__init__(model, corpus, phase)
			self.time_similarity = corpus.time_similarity
			self.geo_similarity = corpus.geo_similarity

		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			pos = self.data['position'][index]
			user_seq = self.corpus.user_his[feed_dict['user_id']][:pos]
			if self.model.history_max > 0:
				user_seq = user_seq[-self.model.history_max:]
			current_timeslot = self.data['c_pre_timeslot_c'][index]
			time_idx = [idx for idx,f in enumerate(self.corpus.context_feature_names) if f=='c_pre_timeslot_c']
			feed_dict['history_time_sim'] = np.array([self.time_similarity[current_timeslot, 
									  x[2][time_idx[0]]] for x in user_seq])
			current_loc = self.data['c_pre_loc_c'][index]
			loc_idx = [idx for idx,f in enumerate(self.corpus.context_feature_names) if f=='c_pre_loc_c']
			feed_dict['history_geo_sim'] = np.array([self.geo_similarity[current_loc, 
									  x[2][loc_idx[0]]] for x in user_seq])
			feed_dict['history_session_id'] = np.array([x[3] for x in user_seq ])
			feed_dict['short_lengths'] = (feed_dict['history_session_id']==feed_dict['history_session_id'].max()).sum() 
			for key in feed_dict:
				if 'history' in key:
					feed_dict[key] = feed_dict[key][::-1].copy()
			return feed_dict