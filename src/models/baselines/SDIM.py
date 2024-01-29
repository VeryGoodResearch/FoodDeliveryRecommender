import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from models.BaseModel import MultiContextSeqModel, MultiContextModel
from utils.layers import MultiHeadTargetAttention, MLP_Block
from pandas.core.common import flatten
import torch.nn.functional as F
import logging

class SDIM(MultiContextSeqModel):
	runner = 'ExpReRunner'
	extra_log_args = ['train_exploration','test_exploration']


	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--dnn_hidden_units',type=str,default='[128,64]')
		parser.add_argument('--dnn_activations',type=str,default='ReLU')
		parser.add_argument('--attention_dim',type=int,default=64)
		parser.add_argument('--use_qkvo',type=int,default=1)
		parser.add_argument('--num_heads',type=int,default=1)
		parser.add_argument('--use_scale',type=int,default=1)
		parser.add_argument('--attention_dropout',type=float,default=0)
		parser.add_argument('--reuse_hash',type=int,default=1)
		parser.add_argument('--num_hashes',type=int,default=1)
		parser.add_argument('--hash_bits',type=int,default=4)
		parser.add_argument('--net_dropout',type=float,default=0)
		parser.add_argument('--batch_norm',type=int,default=0)
		parser.add_argument('--short_target_field',type=str,default='["item","situation"]',
					  help="select from item (will include id), item id, and situation.")
		parser.add_argument('--short_sequence_field',type=str,default='["item","situation"]')
		parser.add_argument('--long_target_field',type=str,default='["item","situation"]')
		parser.add_argument('--long_sequence_field',type=str,default='["item","situation"]')
		parser.add_argument('--output_sigmoid',type=int,default=0)
		parser.add_argument('--group_attention',type=int,default=1)
		parser.add_argument('--all_group_one',type=int,default=0)
		parser.add_argument('--short_history_max',type=int,default=10)
		return MultiContextSeqModel.parse_model_args(parser)

	def get_feature_from_field(self, field_list,status="target"):
		feature_list = []
		for field in field_list:
			if field == "item":
				features = self.item_mh_features + self.item_nu_features
			elif field == "situation":
				features = self.ctx_pre_mh_features + self.ctx_pre_nu_features
			else:
				logging.info("Field %s not defined!"%(field))
				continue
			if status == 'seq':
				features = ['his_'+f for f in features]
			if self.group_attention:
				feature_list.append(tuple(features))
			else:
				feature_list += features 
		if self.all_group_one:
			feature_list_new = []
			for f in feature_list:
				if type(f)==tuple:
					feature_list_new += list(f)
				else:
					feature_list_new.append(f)
			return [tuple(feature_list_new)]
		return feature_list


	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.embedding_dim = args.emb_size
		self.reuse_hash = args.reuse_hash
		self.num_hashes = args.num_hashes
		self.hash_bits = args.hash_bits
		self.group_attention = args.group_attention
		self.all_group_one = args.all_group_one

		self.short_history_max = args.short_history_max	
		self.ctx_pre_mh_features = [f for f in corpus.context_feature_names if 
									 f.endswith("_c") and f.startswith('c_pre')]
		self.ctx_pre_mh_feature_dim = sum([corpus.feature_max[f] for f in self.ctx_pre_mh_features])
		self.ctx_pre_mh_feature_num = len(self.ctx_pre_mh_features)
		self.ctx_pre_nu_features =[f for f in corpus.context_feature_names if 
									 f.endswith("_f") and f.startswith('c_pre')]
		self.ctx_pre_nu_feature_num = len(self.ctx_pre_nu_features)
		self.item_mh_features = [f for f in corpus.item_feature_names if f[-2:]=='_c']
		if self.include_id:
			self.item_mh_features.append('item_id')
		self.item_mh_feature_num = len(self.item_mh_features)
		self.item_nu_features = [f for f in corpus.item_feature_names if f[-2:]=='_f']
		self.item_nu_feature_num = len(self.item_nu_features)
		self.feature_max = corpus.feature_max
		self.user_num = corpus.n_users
		self.short_target_field = self.get_feature_from_field(eval(args.short_target_field))
		self.short_sequence_field = self.get_feature_from_field(eval(args.short_sequence_field),"seq")
		self.long_target_field = self.get_feature_from_field(eval(args.long_target_field))
		self.long_sequence_field = self.get_feature_from_field(eval(args.long_sequence_field),"seq")

		assert len(self.short_target_field) == len(self.short_sequence_field) \
			   and len(self.long_target_field) == len(self.long_sequence_field), \
			   "Config error: target_field mismatches with sequence_field."
	
		self._define_params(args)
		self.apply(self.init_weights)
	
	def _define_params(self,args):
		# embeddings
		self.embedding_dict = nn.ModuleDict()
		for f in self.ctx_pre_mh_features + self.ctx_pre_nu_features + self.item_mh_features + self.item_nu_features:
			self.embedding_dict[f] = nn.Embedding(self.feature_max[f],self.embedding_dim) if f.endswith('_c') or f=='item_id' else\
				nn.Linear(1,self.embedding_dim)
		self.item_embsize = (self.item_mh_feature_num + self.item_nu_feature_num)*self.embedding_dim
		self.ctx_embsize = (self.ctx_pre_mh_feature_num+self.ctx_pre_nu_feature_num)*self.embedding_dim
		self.feature_dim = self.item_embsize + self.ctx_embsize + self.embedding_dim
		
		self.powers_of_two = nn.Parameter(torch.tensor([2.0 ** i for i in range(self.hash_bits)]), 
										  requires_grad=False)
		self.short_attention = nn.ModuleList()
		for target_field in self.short_target_field:
			if type(target_field) == tuple:
				input_dim = self.embedding_dim * len(target_field)
			else:
				input_dim = self.embedding_dim
			self.short_attention.append(MultiHeadTargetAttention(
				input_dim, args.attention_dim, args.num_heads,
				args.attention_dropout, args.use_scale, args.use_qkvo,
			))
		self.random_rotations = nn.ParameterList()
		for target_field in self.long_target_field:
			if type(target_field) == tuple:
				input_dim = self.embedding_dim * len(target_field)
			else:
				input_dim = self.embedding_dim
			self.random_rotations.append(nn.Parameter(torch.randn(input_dim,
								self.num_hashes, self.hash_bits), requires_grad=False))

		self.item_embsize = (self.item_mh_feature_num + self.item_nu_feature_num)*self.embedding_dim
		self.ctx_embsize = (self.ctx_pre_mh_feature_num+self.ctx_pre_nu_feature_num)*self.embedding_dim
		self.long_seq_embsize = len(list(flatten(self.long_sequence_field)))*self.embedding_dim
		self.short_seq_embsize = len(list(flatten(self.short_sequence_field)))*self.embedding_dim
		self.output_activation = self.get_output_activation(args.output_sigmoid)
		self.dnn = MLP_Block(input_dim=self.item_embsize+self.ctx_embsize+self.long_seq_embsize+self.short_seq_embsize,
							 output_dim=1,
							 hidden_units=eval(args.dnn_hidden_units),
							 hidden_activations=args.dnn_activations,
							 output_activation=self.output_activation, 
							 dropout_rates=args.net_dropout,
							 batch_norm=args.batch_norm)

	def forward(self, feed_dict):
		feature_emb_dict = self.get_embeddings(feed_dict)
		lengths = feed_dict['lengths']
		mask = torch.arange(feed_dict['history_items'].shape[1])[None,:].to(self.device) < lengths[:,None] # B * h
		include_features = []
		# short interest attention
		for idx, (target_field, sequence_field) in enumerate(zip(self.short_target_field, 
																 self.short_sequence_field)):
			target_emb = self.concat_embedding(target_field, feature_emb_dict) # batch * item num * embedding
			sequence_emb = self.concat_embedding(sequence_field, feature_emb_dict)
			target_emb_flatten = target_emb.view(-1,target_emb.size(-1))
			sequence_emb_flatten = sequence_emb.unsqueeze(1).repeat(1,target_emb.size(1),1,1).view(
	   					-1,sequence_emb.size(1),sequence_emb.size(2))
			mask_flatten = mask.unsqueeze(1).repeat(1,target_emb.size(1),1).view(-1,sequence_emb.size(1))
			# seq_field = list(flatten([sequence_field]))[0] # flatten nested list to pick the first field
			# mask = X[seq_field].long() != 0 # padding_idx = 0 required in input data
			short_interest_emb_flatten = self.short_attention[idx](target_emb_flatten, sequence_emb_flatten, mask_flatten)
			short_interest_emb = short_interest_emb_flatten.view(target_emb.shape)
			for field, field_emb in zip(list(flatten([sequence_field])),
										short_interest_emb.split(self.embedding_dim, dim=-1)):
				feature_emb_dict[field+'_short'] = field_emb
				include_features.append(field+'_short')
		# long interest attention
		for idx, (target_field, sequence_field) in enumerate(zip(self.long_target_field, 
																 self.long_sequence_field)):
			target_emb = self.concat_embedding(target_field, feature_emb_dict)
			sequence_emb = self.concat_embedding(sequence_field, feature_emb_dict)
			target_emb_flatten = target_emb.view(-1,target_emb.size(-1))
			sequence_emb_flatten = sequence_emb.unsqueeze(1).repeat(1,target_emb.size(1),1,1).view(
	   					-1,sequence_emb.size(1),sequence_emb.size(2))
			long_interest_emb_flatten = self.lsh_attentioin(self.random_rotations[idx], 
													target_emb_flatten, sequence_emb_flatten)
			long_interest_emb = long_interest_emb_flatten.view(target_emb.shape)
			for field, field_emb in zip(list(flatten([sequence_field])),
										long_interest_emb.split(self.embedding_dim, dim=-1)):
				feature_emb_dict[field+'_long'] = field_emb
				include_features.append(field+'_long')
		feature_emb = []
		include_features += self.item_mh_features + self.item_nu_features + self.ctx_pre_mh_features + self.ctx_pre_nu_features
		for f in sorted(include_features):
			feature_emb.append(feature_emb_dict[f])
		feature_emb = torch.cat(feature_emb,dim=-1)

		batch_size, item_num, emb_dim = feature_emb.shape
		y_pred = self.dnn(feature_emb.view(-1,emb_dim)).view(batch_size, item_num, -1).squeeze(-1)
		return_dict = {"prediction": y_pred, "candidate_num":feed_dict['candidates']}
		return return_dict

	def get_embeddings(self, feed_dict):
		feature_emb_dict = dict()
		for f_all in self.ctx_pre_mh_features + self.ctx_pre_nu_features + self.item_mh_features + self.item_nu_features:
			for f in [f_all, 'his_'+f_all]:
				if f.endswith('_c') or f_all=='item_id':
					feature_emb_dict[f] = self.embedding_dict[f_all](feed_dict[f])
				else:
					feature_emb_dict[f] = self.embedding_dict[f_all](feed_dict[f].float().unsqueeze(-1))
		return feature_emb_dict


	def concat_embedding(self, field, feature_emb_dict):
		if type(field) == tuple:
			emb_list = [feature_emb_dict[f] for f in field]
			return torch.cat(emb_list, dim=-1)
		else:
			return feature_emb_dict[field]

	def lsh_attentioin(self, random_rotations, target_item, history_sequence):
		if not self.reuse_hash:
			random_rotations = torch.randn(target_item.size(1), self.num_hashes, 
										   self.hash_bits, device=target_item.device)
		target_bucket = self.lsh_hash(history_sequence, random_rotations)
		sequence_bucket = self.lsh_hash(target_item.unsqueeze(1), random_rotations)
		bucket_match = (sequence_bucket - target_bucket).permute(2, 0, 1) # num_hashes x B x seq_len
		collide_mask = (bucket_match == 0).float()
		hash_index, collide_index = torch.nonzero(collide_mask.flatten(start_dim=1), as_tuple=True)
		offsets = collide_mask.sum(dim=-1).long().flatten().cumsum(dim=0)
		attn_out = F.embedding_bag(collide_index, history_sequence.view(-1, target_item.size(1)), 
								   offsets, mode='sum') # (num_hashes x B) x d
		attn_out = attn_out.view(self.num_hashes, -1, target_item.size(1)).mean(dim=0) # B x d
		return attn_out

	def lsh_hash(self, vecs, random_rotations):
		""" See the tensorflow-lsh-functions for reference:
			https://github.com/brc7/tensorflow-lsh-functions/blob/main/lsh_functions.py
			
			Input: vecs, with shape B x seq_len x d
			Output: hash_bucket, with shape B x seq_len x num_hashes
		"""
		rotated_vecs = torch.einsum("bld,dht->blht", vecs, random_rotations) # B x seq_len x num_hashes x hash_bits
		hash_code = torch.relu(torch.sign(rotated_vecs))
		hash_bucket = torch.matmul(hash_code, self.powers_of_two.unsqueeze(-1)).squeeze(-1)
		return hash_bucket
		
	def get_output_activation(self, output_sigmoid):
		if output_sigmoid:
			return nn.Sigmoid()
		else:
			return nn.Identity()

	class Dataset(MultiContextModel.Dataset):
		def __init__(self, model, corpus, phase):
			super().__init__(model, corpus, phase)
			self.remain_features =  model.ctx_pre_mh_features + model.ctx_pre_nu_features\
	   				+ model.item_mh_features + model.item_nu_features
			self.sample_for_train = model.sample_for_train
			self.include_id = model.include_id
			self.train_exploration = model.train_exploration
			self.test_exploration = model.test_exploration
			self.train_repeat = model.train_repeat
			self.test_repeat = model.test_repeat
			self.phase = phase
			idx_select = []
			for idx in range(len(self.data['position'])):
				if self.sample_for_train==0: # training set not sample negative, i.e., train and test on repeat samples
					neg_items = self.data['neg_items'][idx]
					if (len(neg_items)==1 and neg_items[0] == -1) or self.data['position'][idx]==0: # exploration samples
						continue
					if phase == 'train' and len(neg_items) == 0:
						continue
				else:
					if self.data['position'][idx]==0: # delete first sample for the user
						continue
				if phase=='train' and self.train_exploration and self.data['c_post_repeat_vendor_c'][idx]>0: # only retain exploration interactions in training set
					continue
				if phase!='train' and self.test_exploration and self.data['c_post_repeat_vendor_c'][idx]>0: # only retain exploration interactions in test
					continue
				if phase=='train' and self.train_repeat and self.data['c_post_repeat_vendor_c'][idx]==0: # only retain repeat interactions in training set
					continue
				if phase!='train' and self.test_repeat and self.data['c_post_repeat_vendor_c'][idx]==0: # only retain repeat interactions in test
					continue
				idx_select.append(idx)
			idx_select = np.array(idx_select)
			logging.info('%s data: %d/%d'%(phase, len(idx_select),len(self.data['position'])))
			for key in self.data:
				self.data[key] = np.array(self.data[key])[idx_select]
			self.data['pos_num'] = np.ones(len(self.data['position'])).astype(int)
			if 'neg_items' in self.data:
				self.data['neg_num'] = np.array([len(x) for x in self.data['neg_items']]).astype(int)
			else:
				self.data['neg_num'] = np.array([self.model.num_neg]*len(self.data['position'])).astype(int)

		def actions_before_epoch(self):
			if self.sample_for_train:
				neg_items = np.random.randint(1, self.corpus.n_items, size=(len(self), self.model.num_neg))
				for i, iid in enumerate(self.data['item_id']): # negative items just need to be different from the current pos
					for j in range(self.model.num_neg):
						while neg_items[i][j] == iid:
							neg_items[i][j] = np.random.randint(1, self.corpus.n_items)
				self.data['neg_items'] = neg_items
				self.data['neg_num'] = (np.ones(len(neg_items))*self.model.num_neg).astype(int)	

		def _get_feed_dict(self, index):
			user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
			neg_items = self.data['neg_items'][index]
			feed_dict = {
				'user_id': user_id,
				'item_id': np.concatenate([[target_item],neg_items]).astype(int),
				'candidates': len(neg_items)+1,
				'pos_num':1, 'neg_num':len(neg_items),
				'repeat_time':self.data['c_post_repeat_vendor_c'][index],
			}
			pos = self.data['position'][index]
			user_seq = self.corpus.user_his[feed_dict['user_id']][:pos]
			if self.model.history_max > 0:
				user_seq = user_seq[-self.model.history_max:]
			feed_dict['history_items'] = np.array([x[0] for x in user_seq])
			feed_dict = self.get_user_features(feed_dict)
			feed_dict = self.get_item_features(feed_dict)
			feed_dict = self.get_context_features(feed_dict,index, user_seq)
			feed_dict['lengths'] = len(feed_dict['history_items'])

			return feed_dict

		def get_item_features(self, feed_dict):
			item_mh_fnames = [f for f in self.corpus.item_feature_names if f[-2:]=='_c']
			item_nu_fnames = [f for f in self.corpus.item_feature_names if f[-2:]=='_f']
			item_num = 1 if type(feed_dict['item_id']) == int else len(feed_dict['item_id'])
			for f in item_mh_fnames + item_nu_fnames:
				feed_dict[f] = max(self.corpus.item_features[feed_dict['item_id']][f],0) if (item_num == 1 and type(feed_dict['item_id'])==int) else\
					np.array([max(self.corpus.item_features[iid][f],0) for iid in feed_dict['item_id']])
				feed_dict['his_'+f] = np.array([max(self.corpus.item_features[iid][f],0) for iid in feed_dict['history_items']])
			if self.include_id:
				feed_dict['his_item_id'] = feed_dict['history_items']	
			return feed_dict

		def get_context_features(self, feed_dict, index, user_seq):
			item_num = 1 if type(feed_dict['item_id']) == int else len(feed_dict['item_id'])
			context_pre_mh_fnames = [(idx,f) for idx,f in enumerate(self.corpus.context_feature_names) if f[-2:]=='_c' and f.startswith('c_pre')]
			context_pre_nu_fnames = [(idx,f) for idx,f in enumerate(self.corpus.context_feature_names) if f[-2:]=='_f' and f.startswith('c_pre')]
			for idx,f in context_pre_mh_fnames + context_pre_nu_fnames:
				feed_dict[f] = self.data[f][index] if (item_num == 1 and type(feed_dict['item_id'])==int) else np.array([self.data[f][index] for i in range(item_num)])
				feed_dict['his_'+f] = np.array([x[2][idx] for x in user_seq])
			return feed_dict
 