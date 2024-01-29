import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
import pandas as pd

from models.BaseModel import MultiContextSeqModel

class ReCANet(MultiContextSeqModel):
	# runner = 'ImpressionRunner'
	runner = 'ExpReRunner'
	extra_log_args = ['history_max','history_w','sample_for_train']
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--u_emb_size', type=int, default=64,
							help='Size of user embedding vectors.')
		parser.add_argument('--i_emb_size', type=int, default=64,
							help='Size of item embedding vectors.')
		parser.add_argument('--m_emb_size', type=int, default=64,
							help='Size of concat embedding vectors.')
		parser.add_argument('--history_w',type=int,default=10)
		parser.add_argument('--layers', type=str, default='[64,64]',
							help="Size of each layer.")
		return MultiContextSeqModel.parse_model_args(parser)	
	
	def __init__(self, args, corpus):
		super().__init__(args, corpus)

		self.u_emb_size = args.u_emb_size
		self.i_emb_size = args.i_emb_size
		self.m_emb_size = args.m_emb_size
		self.history_w = args.history_w
		self.hidden_size = args.m_emb_size
		self.layers = eval(args.layers)

		self._define_params()
		self.apply(self.init_weights)
	
	def _define_params(self):
		self.item_embeddings = nn.Embedding(self.item_num,self.i_emb_size)
		self.user_embeddings = nn.Embedding(self.user_num,self.u_emb_size)
		self.ui_concat = nn.Linear(self.u_emb_size+self.i_emb_size, self.m_emb_size)
		self.uih_concat = nn.Linear(self.m_emb_size+1,self.m_emb_size)
		self.lstm = nn.LSTM(input_size=self.m_emb_size, hidden_size=self.hidden_size, num_layers=2,
					   dropout=self.dropout, batch_first=True)
		self.prediction_layer = nn.ModuleList()
		pre_size = self.hidden_size
		for size in self.layers:
			self.prediction_layer.append(nn.Linear(pre_size,size))
			self.prediction_layer.append(nn.ReLU())
			self.prediction_layer.append(nn.Dropout())
			pre_size = size
		self.prediction_layer.append(nn.Linear(pre_size,1))


	def forward(self, feed_dict):
		lengths = feed_dict['lengths']
		item_ids = feed_dict['item_id']
		user_id = feed_dict['user_id']
		history_items = feed_dict['history_items']
		batch_size, candidate_num = item_ids.shape
		history_size = history_items.shape[-1]

		matching = history_items[:,None,:] == item_ids[:,:,None]
		indices = torch.nonzero(matching==1, as_tuple=False)
		history_index = -torch.ones_like(matching).long()
		history_index[indices[:,0],indices[:,1],indices[:,2]] = indices[:,2]
		history_index = torch.sort(history_index,dim=-1,descending=True)[0]
		history_index = history_index[:,:,:self.history_w]
		interval_mask = history_index>-1
		history_interval = ((lengths[:,None,None]-history_index)*interval_mask).unsqueeze(-1) # B*c*history*1

		item_emb = self.item_embeddings(item_ids) # B * c * i emb
		user_emb = self.user_embeddings(user_id) # B * u emb
		ui_emb = fn.relu(self.ui_concat(torch.cat([user_emb.unsqueeze(1).repeat(1,candidate_num,1),item_emb],
											dim=-1))) # B * c * m
		ui_c = fn.relu(self.uih_concat(torch.cat([ui_emb.unsqueeze(-2).repeat(1,1,min(history_size,self.history_w),1),
								history_interval],dim=-1))) # B * c * len * m

		ui_c_flatten = ui_c.flatten(start_dim=0,end_dim=1) # (B*c) * len * m
		interval_lengths_flatten = interval_mask.sum(dim=-1).flatten().clamp(1)

		# sort and pack
		sort_his_lengths, sort_idx = torch.topk(interval_lengths_flatten, k=len(interval_lengths_flatten))
		sort_his_vectors = ui_c_flatten.index_select(dim=0, index=sort_idx)
		history_packed = torch.nn.utils.rnn.pack_padded_sequence(
			sort_his_vectors, sort_his_lengths.cpu(), batch_first=True)
		# RNN
		output, (hidden, c) = self.lstm(history_packed, None)
		# Unsort
		unsort_idx = torch.topk(sort_idx, k=len(interval_lengths_flatten), largest=False)[1]
		rnn_vector = hidden[-1].index_select(dim=0, index=unsort_idx)

		rnn_vector = rnn_vector.view(batch_size,candidate_num,-1)
	
		outputs = rnn_vector
		for layer in self.prediction_layer:
			outputs = layer(outputs)
		prediction = outputs.squeeze(-1)
		return {'prediction':prediction, 'candidate_num':feed_dict['candidates']}


	class Dataset(MultiContextSeqModel.Dataset):
		def _get_feed_dict(self, index):
			feed_dict =  super()._get_feed_dict(index)
			# feed_dict['history_interval'] = []
			# for idx
			return feed_dict