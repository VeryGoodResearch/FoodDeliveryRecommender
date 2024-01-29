import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from models.BaseModel import MultiContextSeqModel
from utils import layers
import logging
from tqdm import tqdm

class Ensemble(MultiContextSeqModel):
	runner = 'EnsembleRunner'
	reader = 'ContextScoreReader'
	extra_log_args = ['base_model1','base_model2','loss_alpha']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--encoder',type=str,default='BERT4Rec',
				help='A sequence encoder for intent prediction.')
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--num_heads', type=int, default=1,
							help='Number of attention heads.')
		parser.add_argument('--num_layers', type=int, default=1,
							help='Number of self-attention layers.')
		parser.add_argument('--cross_attn_qsize',type=int,default=32,help='Embedding size for cross-attention query.')
		parser.add_argument('--cross_attention',type=int,default=1,help='Using cross-attention structure or direct attention.')
		parser.add_argument('--loss_alpha',type=float,default=1)
		return MultiContextSeqModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args,corpus)
		self.embedding_dim = args.emb_size

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

		self.head_num = args.num_heads
		self.layer_num = args.num_layers

		self.cross_attention = args.cross_attention
		self.cross_attn_qsize = args.cross_attn_qsize
		self.encoder_name = args.encoder
		self.model_num = 2
		self.intent_num = 2

		self.loss_alpha = args.loss_alpha

		self._define_params(args)
		self.apply(self.init_weights)


	def _define_params(self,args):
		# embeddings
		self.intent_embedding = nn.Embedding(2, self.embedding_dim)
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
		self.ctx2emb_linear = nn.Linear(self.ctx_embsize,self.embedding_dim)

		# self attention
		self.score_embeddings = nn.Linear(self.model_num, self.embedding_dim)
		self.s_attn_head = layers.MultiHeadAttention(self.embedding_dim, self.head_num, bias=False)
		self.s_W1 = nn.Linear(self.embedding_dim,self.embedding_dim)
		self.s_W2 = nn.Linear(self.embedding_dim,self.embedding_dim)
		self.dropout_layer = nn.Dropout(self.dropout)
		self.s_layer_norm = nn.LayerNorm(self.embedding_dim)

		# cross attention
		self.act_func = nn.ReLU
		if self.cross_attention:
			self.intent_score_attention = layers.CrossAtt(input_qsize=self.embedding_dim, input_vsize=self.embedding_dim, 
								query_size=self.embedding_dim,key_size=self.embedding_dim,value_size=self.embedding_dim)
		else:
			self.intent_score_embeddings = nn.Sequential(
						nn.Linear(self.embedding_dim, self.cross_attn_qsize),
						self.act_func(),
						nn.Linear(self.cross_attn_qsize, self.embedding_dim, bias=False)
						)

		self.weight_embeddings = nn.Linear(self.embedding_dim*3, self.model_num)

		# encoder
		self.intent_pred_size = self.item_embsize + self.ctx_embsize 
		if self.encoder_name == 'GRU4Rec':
			self.encoder = GRU4RecEncoder(self.intent_pred_size,hidden_size=self.intent_pred_size,out_size=self.embedding_dim)
			self.intent_encoder = GRU4RecEncoder(self.embedding_dim,hidden_size=self.embedding_dim)
		elif self.encoder_name == 'BERT4Rec':
			self.encoder = BERT4RecEncoder(self.intent_pred_size,self.history_max,num_layers=2,num_heads=2)
			self.intent_encoder = BERT4RecEncoder(self.embedding_dim,self.history_max,num_layers=2,num_heads=2)
			self.encoder_output = nn.Linear(self.intent_pred_size, self.embedding_dim)
		else:
			raise ValueError('Invalid sequence encoder.')
		self.intent_hidden_layer = nn.Linear( self.embedding_dim*4, self.embedding_dim)
		self.intent_pred_layer = nn.Linear(self.embedding_dim, self.intent_num)

		# loss
		self.intent_loss = nn.BCELoss(reduction='none')

	def forward(self, feed_dict):
		# predict intents
		intent, intent_emb = self.predict_intent(feed_dict)
		# predict ensemble
		weights, ens_scores = self.predict_ensemble(feed_dict,intent_emb)

		repeat_candidates = (feed_dict['scores'][:,:,1]==0).float()
		out_dict = {"weights":weights,"prediction":ens_scores,"intents":intent,"candidate_num":feed_dict['candidates'],
              "repeat":feed_dict['repeat_time']>0,'repeat_candidate':repeat_candidates}
		return out_dict

	def predict_intent(self,feed_dict):
		# item embeddings
		history_items_mh = self.item_mh_embeddings(feed_dict['history_item_mh_features'])
		if self.item_nu_feature_num>0:
			history_items_nu = feed_dict['history_item_nu_features'].float()
			history_items_nu_emb = []
			for i, embedding_layer in enumerate(self.item_nu_embeddings):
				history_items_nu_emb.append(embedding_layer(history_items_nu[:,:,i].unsqueeze(-1)))
			history_items_nu_emb = torch.stack(history_items_nu_emb,dim=-2)
			history_items = torch.cat([history_items_mh,history_items_nu_emb],dim=-2) # B * h * if num * emb
		else:
			history_items = history_items_mh
		history_items = history_items.flatten(start_dim=-2)

		# context embeddings
		lengths = feed_dict['lengths']
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
			his_situation_vectors = torch.cat([his_pre_ctx_mh,his_pre_ctx_nu_emb],dim=-2)
			situation_vectors = torch.cat([pre_ctx_mh,pre_ctx_nu_emb],dim=-2)
		else:
			his_situation_vectors = his_pre_ctx_mh
			situation_vectors = pre_ctx_mh
		his_situation_vectors = his_situation_vectors.flatten(start_dim=-2)
		situation_vectors = self.ctx2emb_linear(situation_vectors.flatten(start_dim=-2).squeeze(dim=1))

		his_intents = feed_dict['his_repeat'].long()
		his_intent_emb = self.intent_embedding(his_intents)

		# get history encoding	
		his_embedding = torch.cat([his_situation_vectors,history_items],dim=-1)
		his_vector = self.encoder(his_embedding,lengths)
		intent_emb = self.intent_encoder(his_intent_emb, lengths)
		if self.encoder_name == 'BERT4Rec':
			his_vector = self.encoder_output(his_vector)

		# other current embeddings
		user_emb = self.user_embedding(feed_dict['user_id'])
		
		# current_embeddings = torch.cat([situation_vectors,user_emb],dim=-1) # batch size * embedding dim
		current_embeddings = torch.cat([intent_emb,situation_vectors,user_emb],dim=-1) # batch size * embedding dim

		# predict intent probability
		hidden_intents = self.intent_hidden_layer(torch.cat([current_embeddings,his_vector],dim=-1))
		pred_intents = self.intent_pred_layer(hidden_intents).softmax(dim=-1)

		return pred_intents, hidden_intents # b * 2

	def predict_ensemble(self,feed_dict,intent_emb):
		# load data
		score_list = feed_dict['scores'].float() # b * item num * model num
		score_max = score_list.max(dim=1)[0]
		score_min = score_list.min(dim=1)[0]
		score_norm = (score_list - score_min[:,None,:])/(score_max-score_min)[:,None,:]

		# intent embedding
		h_int = intent_emb.unsqueeze(dim=1)

		# user embedding
		user_ids = (feed_dict['user_id'])
		h_u = F.relu(self.user_embedding(user_ids)) # batch * u_emb_size

		# self attention
		h_s = self.score_embeddings(score_list)
  
		# cross-attention
		if self.cross_attention:
			score_xatt, score_xatt_w = self.intent_score_attention(h_s,h_int,valid=None, scale=1/torch.sqrt(torch.tensor(self.cross_attn_qsize)),act_v=None)
		else:
			score_intent = self.intent_score_embeddings(h_int)
			score_xatt = torch.mul(h_s,score_intent)

		# to weight
		h_u = h_u.unsqueeze(dim=1).repeat(1,score_list.size(1),1)
		all_xatt = torch.cat([score_xatt,h_u,h_int.repeat(1,score_list.size(1),1)],dim=-1)
		weights = self.weight_embeddings(all_xatt).softmax(dim=-1)
		ens_score = torch.mul(weights,score_norm).sum(dim=2)
		
		return weights, ens_score

	def loss_bpr(self, out_dict, labels=None, mean=True):
		prediction = out_dict['prediction']
		neg_mask = (1-out_dict['repeat_candidate'])*out_dict["repeat"][:,None]+(out_dict['repeat_candidate'])*(~out_dict["repeat"])[:,None]
		neg_mask = neg_mask[:,1:]
		neg_num = neg_mask.sum(dim=-1)
		neg_mask[:,0] += (neg_num==0).float()
		pos_pred, neg_pred = prediction[:,0], prediction[:,1:]
		neg_pred = torch.where(neg_mask==1,neg_pred,-torch.tensor(float("Inf")).float().to(self.device))
		neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=-1)
		loss = -((pos_pred[:,None] - neg_pred).sigmoid() * neg_softmax * neg_mask).sum(dim=1).log()
		if mean:
			loss = loss.mean()
		return loss

	def loss(self, out_dict, labels=None):
		rec_loss = self.loss_bpr(out_dict,labels=labels,mean=False)
		pred_intents = out_dict["intents"]
		int_loss = self.intent_loss(pred_intents[:,0], out_dict["repeat"].float())
		loss = (rec_loss*self.loss_alpha + int_loss).mean()
		return loss

	class Dataset(MultiContextSeqModel.Dataset):
		def __init__(self,model,corpus,phase): 
			super().__init__(model, corpus, phase)
			self.scores = corpus.scores[phase]
		# Called after initialization
		def prepare(self):
			if self.model.buffer and self.phase not in  ['dev','train']:
				for i in tqdm(range(len(self)), leave=False, desc=('Prepare ' + self.phase)):
					self.buffer_dict[i] = self._get_feed_dict(i)

		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			feed_dict['scores'] = self.scores[index,:,:]
			pos = self.data['position'][index]
			user_seq = self.corpus.user_his[feed_dict['user_id']][:pos]
			if self.model.history_max > 0:
				user_seq = user_seq[-self.model.history_max:]
			repeat_idx = [idx for idx,f in enumerate(self.corpus.context_feature_names) if f=='c_post_repeat_vendor_c']
			feed_dict['his_repeat'] = np.array([x[2][repeat_idx[0]]>0 for x in user_seq])
			return feed_dict

		def __getitem__(self, index: int) -> dict:
			if self.model.buffer and self.phase not in ['train','dev']:
				return self.buffer_dict[index]
			return self._get_feed_dict(index)

""" Encoder Layers """
class GRU4RecEncoder(nn.Module):
	def __init__(self,emb_size, hidden_size=128,out_size=0):
		super().__init__()
		if out_size == 0:
			out_size = emb_size
		self.rnn = nn.GRU(input_size=emb_size, hidden_size=hidden_size, batch_first=True)
		self.out = nn.Linear(hidden_size, out_size, bias=False)

	def forward(self, seq, lengths):
		# Sort and Pack
		sort_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
		sort_seq = seq.index_select(dim=0, index=sort_idx)
		seq_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_seq, sort_lengths.cpu(), batch_first=True)

		# RNN
		output, hidden = self.rnn(seq_packed, None)

		# Unsort
		sort_rnn_vector = self.out(hidden[-1])
		unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
		rnn_vector = sort_rnn_vector.index_select(dim=0, index=unsort_idx)

		return rnn_vector

class BERT4RecEncoder(nn.Module):
	def __init__(self, emb_size, max_his, num_layers=2, num_heads=2):
		super().__init__()
		self.p_embeddings = nn.Embedding(max_his + 1, emb_size)
		self.transformer_block = nn.ModuleList([
			layers.TransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=num_heads)
			for _ in range(num_layers)
		])

	def forward(self, seq, lengths):
		batch_size, seq_len = seq.size(0), seq.size(1)
		len_range = torch.from_numpy(np.arange(seq_len)).to(seq.device)
		valid_mask = len_range[None, :] < lengths[:, None]

		# Position embedding
		position = len_range[None, :] * valid_mask.long()
		pos_vectors = self.p_embeddings(position)
		seq = seq + pos_vectors

		# Self-attention
		attn_mask = valid_mask.view(batch_size, 1, 1, seq_len)
		for block in self.transformer_block:
			seq = block(seq, attn_mask)
		seq = seq * valid_mask[:, :, None].float()

		his_vector = seq[torch.arange(batch_size), lengths - 1]
		return his_vector