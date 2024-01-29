# -*- coding: UTF-8 -*-
'''
Reference:
	https://github.com/THUwangcy/ReChorus
'''

import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence
from typing import List

from utils import utils
from helpers.BaseReader import BaseReader


class BaseModel(nn.Module):
	reader, runner = None, None  # choose helpers in specific model classes
	extra_log_args = []

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--model_path', type=str, default='',
							help='Model save path.')
		parser.add_argument('--buffer', type=int, default=1,
							help='Whether to buffer feed dicts for dev/test')
		return parser

	@staticmethod
	def init_weights(m):
		if 'Linear' in str(type(m)):
			nn.init.normal_(m.weight, mean=0.0, std=0.01)
			if m.bias is not None:
				nn.init.normal_(m.bias, mean=0.0, std=0.01)
		elif 'Embedding' in str(type(m)):
			nn.init.normal_(m.weight, mean=0.0, std=0.01)

	def __init__(self, args, corpus: BaseReader):
		super(BaseModel, self).__init__()
		self.device = args.device
		self.model_path = args.model_path
		self.buffer = args.buffer
		self.optimizer = None
		self.check_list = list()  # observe tensors in check_list every check_epoch

	"""
	Key Methods
	"""
	def _define_params(self):
		pass

	def forward(self, feed_dict: dict) -> dict:
		"""
		:param feed_dict: batch prepared in Dataset
		:return: out_dict, including prediction with shape [batch_size, n_candidates]
		"""
		pass

	def loss(self, out_dict: dict) -> torch.Tensor:
		pass

	"""
	Auxiliary Methods
	"""
	def customize_parameters(self) -> list:
		# customize optimizer settings for different parameters
		weight_p, bias_p = [], []
		for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
			if 'bias' in name:
				bias_p.append(p)
			else:
				weight_p.append(p)
		optimize_dict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
		return optimize_dict

	def save_model(self, model_path=None):
		if model_path is None:
			model_path = self.model_path
		utils.check_dir(model_path)
		torch.save(self.state_dict(), model_path)
		# logging.info('Save model to ' + model_path[:50] + '...')

	def load_model(self, model_path=None):
		if model_path is None:
			model_path = self.model_path
		self.load_state_dict(torch.load(model_path))
		logging.info('Load model from ' + model_path)

	def count_variables(self) -> int:
		total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
		return total_parameters

	def actions_after_train(self):  # e.g., save selected parameters
		pass

	"""
	Define Dataset Class
	"""
	class Dataset(BaseDataset):
		def __init__(self, model, corpus, phase: str):
			self.model = model  # model object reference
			self.corpus = corpus  # reader object reference
			self.phase = phase  # train / dev / test

			self.buffer_dict = dict()
			self.data = corpus.data_df[phase].to_dict('list')
			# ↑ DataFrame is not compatible with multi-thread operations

		def __len__(self):
			if type(self.data) == dict:
				for key in self.data:
					return len(self.data[key])
			return len(self.data)

		def __getitem__(self, index: int) -> dict:
			if self.model.buffer and self.phase != 'train':
				return self.buffer_dict[index]
			return self._get_feed_dict(index)

		# ! Key method to construct input data for a single instance
		def _get_feed_dict(self, index: int) -> dict:
			pass

		# Called after initialization
		def prepare(self):
			if self.model.buffer and self.phase != 'train':
				for i in tqdm(range(len(self)), leave=False, desc=('Prepare ' + self.phase)):
					self.buffer_dict[i] = self._get_feed_dict(i)

		# Called before each training epoch (only for the training dataset)
		def actions_before_epoch(self):
			pass

		# Collate a batch according to the list of feed dicts
		def collate_batch(self, feed_dicts: List[dict]) -> dict:
			feed_dict = dict()
			for key in feed_dicts[0]:
				if isinstance(feed_dicts[0][key], np.ndarray):
					tmp_list = [len(d[key]) for d in feed_dicts]
					if any([tmp_list[0] != l for l in tmp_list]):
						stack_val = np.array([d[key] for d in feed_dicts], dtype=np.object)
					else:
						try:
							stack_val = np.array([d[key] for d in feed_dicts])
						except:
							print('error')
				else:
					stack_val = np.array([d[key] for d in feed_dicts])
				if stack_val.dtype == np.object:  # inconsistent length (e.g., history)
					feed_dict[key] = pad_sequence([torch.from_numpy(x) for x in stack_val], batch_first=True)
				else:
					feed_dict[key] = torch.from_numpy(stack_val)
			feed_dict['batch_size'] = len(feed_dicts)
			feed_dict['phase'] = self.phase
			return feed_dict

class GeneralModel(BaseModel):
	reader, runner = 'BaseReader', 'BaseRunner'

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--num_neg', type=int, default=1,
							help='The number of negative items during training.')
		parser.add_argument('--dropout', type=float, default=0,
							help='Dropout probability for each deep layer')
		parser.add_argument('--test_all', type=int, default=0,
							help='Whether testing on all the items.')
		return BaseModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.user_num = corpus.n_users
		self.item_num = corpus.n_items
		self.num_neg = args.num_neg
		self.dropout = args.dropout
		self.test_all = args.test_all

	def loss(self, out_dict: dict) -> torch.Tensor:
		"""
		BPR ranking loss with optimization on multiple negative samples (a little different now to follow the paper ↓)
		"Recurrent neural networks with top-k gains for session-based recommendations"
		:param out_dict: contain prediction with [batch_size, -1], the first column for positive, the rest for negative
		:return:
		"""
		predictions = out_dict['prediction']
		pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
		neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
		loss = -((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).log().mean()
		# neg_pred = (neg_pred * neg_softmax).sum(dim=1)
		# loss = F.softplus(-(pos_pred - neg_pred)).mean()
		# ↑ For numerical stability, use 'softplus(-x)' instead of '-log_sigmoid(x)'
		return loss

	class Dataset(BaseModel.Dataset):
		def _get_feed_dict(self, index):
			user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
			if self.phase != 'train' and self.model.test_all:
				neg_items = np.arange(1, self.corpus.n_items)
			else:
				neg_items = self.data['neg_items'][index]
			item_ids = np.concatenate([[target_item], neg_items]).astype(int)
			feed_dict = {
				'user_id': user_id,
				'item_id': item_ids
			}
			return feed_dict

		# Sample negative items for all the instances
		def actions_before_epoch(self):
			neg_items = np.random.randint(1, self.corpus.n_items, size=(len(self), self.model.num_neg))
			for i, u in enumerate(self.data['user_id']):
				clicked_set = self.corpus.train_clicked_set[u]  # neg items are possible to appear in dev/test set
				# clicked_set = self.corpus.clicked_set[u]  # neg items will not include dev/test set
				for j in range(self.model.num_neg):
					while neg_items[i][j] in clicked_set:
						neg_items[i][j] = np.random.randint(1, self.corpus.n_items)
			self.data['neg_items'] = neg_items


class MultiContextModel(GeneralModel):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--loss_n',type=str,default='BPR',
							help='Type of loss functions.')
		parser.add_argument('--include_id',type=int,default=1,
							help='Whether add ids in context information.')
		parser.add_argument('--sample_for_train',type=int,default=0,
					  help='Whether do negative sampling for training set. If 0, only train and test on repeat samples')
		parser.add_argument('--train_exploration',type=int,default=0,
                      help='Whether train only on the exploration samples.')
		parser.add_argument('--test_exploration',type=int,default=0,
                      help='Whether test only on the exploration samples.')
		parser.add_argument('--train_repeat',type=int,default=0,
                      help='Whether train only on the repeat samples.')
		parser.add_argument('--test_repeat',type=int,default=0,
                      help='Whether test only on the repeat samples.')
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.loss_n = args.loss_n
		self.include_id = args.include_id
		self.sample_for_train = args.sample_for_train
		self.train_max_pos_item = self.test_max_pos_item = 1
		self.train_max_neg_item = self.test_max_neg_item = 1000
		self.train_exploration = args.train_exploration
		self.test_exploration = args.test_exploration
		self.train_repeat = args.train_repeat
		self.test_repeat = args.test_repeat

	def loss(self, out_dict, labels=None, mean=True):
		prediction = out_dict['prediction']
		candidate_num = out_dict['candidate_num']
		neg_mask = (torch.arange(prediction.shape[1])[None,:].to(self.device) < candidate_num[:,None])[:,1:]
		pos_pred, neg_pred = prediction[:,0], prediction[:,1:]
		neg_pred = torch.where(neg_mask==1,neg_pred,-torch.tensor(float("Inf")).float().to(self.device))
		neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=-1)
		loss = -((pos_pred[:,None] - neg_pred).sigmoid() * neg_softmax * neg_mask).sum(dim=1).log()
		if mean:
			loss = loss.mean()
		return loss

	class Dataset(GeneralModel.Dataset):
		def _convert_multihot(self, fname_list, data):
			base = 0
			for i, feature in enumerate(fname_list):
				data[:,i] += base
				base += self.corpus.feature_max[feature]
			return data

		def get_user_features(self,feed_dict):
			user_mh_fnames = [f for f in self.corpus.user_feature_names if f[-2:]=='_c']
			user_nu_fnames = [f for f in self.corpus.user_feature_names if f[-2:]=='_f']
			if len(user_mh_fnames):
				user_mh_features = [self.corpus.user_features[feed_dict['user_id']][c] for c in user_mh_fnames] 
				if self.include_id:
					user_mh_fnames = ['user_id'] + user_mh_fnames
					user_mh_features = [feed_dict['user_id']] + user_mh_features
				feed_dict['user_mh_features'] = self._convert_multihot(user_mh_fnames, np.array(user_mh_features).reshape(1,-1))
			elif self.include_id:
				feed_dict['user_mh_features'] = feed_dict['user_id'].reshape(-1,1)
			if len(user_nu_fnames):
				feed_dict['user_nu_features'] = np.array([self.corpus.user_features[feed_dict['user_id']][c] for c in user_nu_fnames]).astype(float).reshape(1,-1)
			return feed_dict

		def get_item_features(self, feed_dict):
			item_mh_fnames = [f for f in self.corpus.item_feature_names if f[-2:]=='_c']
			item_nu_fnames = [f for f in self.corpus.item_feature_names if f[-2:]=='_f']
			if len(item_mh_fnames):
				if type(feed_dict['item_id']) == int:
					item_num = 1
					item_mh_features = [max(self.corpus.item_features[feed_dict['item_id']][c],0) for c in item_mh_fnames] 
				else: # multiple items
					item_num = len(feed_dict['item_id'])
					item_mh_features = np.array([[max(self.corpus.item_features[iid][c],0) for c in item_mh_fnames] 
											for iid in feed_dict['item_id'] ])
				if self.include_id:
					item_mh_fnames = ['item_id'] + item_mh_fnames
					if type(feed_dict['item_id']) == int:
						item_mh_features = [feed_dict['item_id']] + item_mh_features
					else:
						item_mh_features = np.concatenate([feed_dict['item_id'].reshape(-1,1),item_mh_features],axis=-1)
				feed_dict['item_mh_features'] = self._convert_multihot(item_mh_fnames, np.array(item_mh_features).reshape(item_num,-1))
			elif self.include_id:
				feed_dict['item_mh_features'] = feed_dict['item_id'].reshape(-1,1)
			if len(item_nu_fnames):
				if type(feed_dict['item_id']) == int:
					feed_dict['item_nu_features'] = np.array([max(self.corpus.item_features[feed_dict['item_id']][c],0)
											   for c in item_nu_fnames]).astype(float).reshape(item_num,-1)
				else:
					feed_dict['item_nu_features'] = np.array([[max(self.corpus.item_features[iid][c],0) for c in item_nu_fnames]
											   for iid in feed_dict['item_id']]).astype(float)
			return feed_dict

		def get_item_seq_features(self, feed_dict):
			item_mh_fnames = [f for f in self.corpus.item_feature_names if f[-2:]=='_c']
			item_nu_fnames = [f for f in self.corpus.item_feature_names if f[-2:]=='_f']
			if len(item_mh_fnames):
				item_mh_features_history = np.array([[max(self.corpus.item_features[iid][c],0)
								 for c in item_mh_fnames] for iid in feed_dict['history_items']])
				if self.include_id:
					item_mh_fnames = ['item_id'] + item_mh_fnames
					his_item_ids = feed_dict['history_items'].reshape(-1,1)
					item_mh_features_history = np.concatenate([his_item_ids,item_mh_features_history],axis=1)
				feed_dict['history_item_mh_features'] = self._convert_multihot(item_mh_fnames, item_mh_features_history)
			elif self.include_id:
				feed_dict['history_item_mh_features'] = feed_dict['history_items'].reshape(-1,1)
			if len(item_nu_fnames):
				feed_dict['history_item_nu_features'] = np.array([[max(self.corpus.item_features[iid][c],0)
							for c in item_nu_fnames] for iid in feed_dict['history_items']]).astype(float)
			return feed_dict

		def get_context_features(self, feed_dict, index):
			context_pre_mh_fnames = [f for f in self.corpus.context_feature_names if f[-2:]=='_c' and f.startswith('c_pre')]
			context_pre_nu_fnames = [f for f in self.corpus.context_feature_names if f[-2:]=='_f' and f.startswith('c_pre')]
			if len(context_pre_mh_fnames):
				context_pre_mh_features = [self.data[c][index] for c in context_pre_mh_fnames] 
				feed_dict['context_pre_mh_features'] = self._convert_multihot(context_pre_mh_fnames, np.array(context_pre_mh_features).reshape(1,-1))
			if len(context_pre_nu_fnames):
				feed_dict['context_pre_nu_features'] = np.array([self.data[c][index] for c in context_pre_nu_fnames]).astype(float).reshape(1,-1)
			context_post_mh_fnames = [f for f in self.corpus.context_feature_names if f[-2:]=='_c' and f.startswith('c_post')]
			context_post_nu_fnames = [f for f in self.corpus.context_feature_names if f[-2:]=='_f' and f.startswith('c_post')]
			if len(context_post_mh_fnames):
				context_post_mh_features = [self.data[c][index] for c in context_post_mh_fnames] 
				feed_dict['context_post_mh_features'] = self._convert_multihot(context_post_mh_fnames, np.array(context_post_mh_features).reshape(1,-1))
			if len(context_post_nu_fnames):
				feed_dict['context_post_nu_features'] = np.array([self.data[c][index] for c in context_post_nu_fnames]).astype(float).reshape(1,-1)
			return feed_dict
 
		def get_context_seq_features(self, feed_dict, index):
			pos = self.data['position'][index]
			user_seq = self.corpus.user_his[feed_dict['user_id']][:pos]
			if self.model.history_max > 0:
				user_seq = user_seq[-self.model.history_max:]
			feed_dict['history_items'] = np.array([x[0] for x in user_seq])
			context_pre_mh_fnames = [(idx,f) for idx,f in enumerate(self.corpus.context_feature_names) if f[-2:]=='_c' and f.startswith('c_pre')]
			context_pre_nu_fnames = [(idx,f) for idx,f in enumerate(self.corpus.context_feature_names) if f[-2:]=='_f' and f.startswith('c_pre')]
			if len(context_pre_mh_fnames):
				history_context_pre_mh_features =  [[x[2][idx] for idx,f in context_pre_mh_fnames] for x in user_seq]
				if len(history_context_pre_mh_features)==0:
					print("error")
				feed_dict['history_context_pre_mh_features'] = self._convert_multihot([i[1] for i in context_pre_mh_fnames], np.array(history_context_pre_mh_features))
			if len(context_pre_nu_fnames):
				feed_dict['history_context_pre_nu_features'] = np.array([[x[2][idx] for idx,f in context_pre_nu_fnames] for x in user_seq]).astype(float)
			context_post_mh_fnames = [(idx,f) for idx,f in enumerate(self.corpus.context_feature_names) if f[-2:]=='_c' and f.startswith('c_post')]
			context_post_nu_fnames = [(idx,f) for idx,f in enumerate(self.corpus.context_feature_names) if f[-2:]=='_f' and f.startswith('c_post')]
			if len(context_post_mh_fnames):
				history_context_post_mh_features =  [[x[2][idx] for idx,f in context_post_mh_fnames] for x in user_seq]
				feed_dict['history_context_post_mh_features'] = self._convert_multihot([i[1] for i in context_post_mh_fnames], np.array(history_context_post_mh_features))
			if len(context_post_nu_fnames):
				feed_dict['history_context_post_nu_features'] = np.array([[x[2][idx] for idx,f in context_post_nu_fnames] for x in user_seq]).astype(float)
			return feed_dict
	
class CTRModel(GeneralModel):
	reader, runner = 'ContextReader', 'BaseRunner'
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--loss_n',type=str,default='BPR',
							help='Type of loss functions.')
		parser.add_argument('--include_id',type=int,default=1,
							help='Whether add ids in context information.')
		parser.add_argument('--label_column',type=str,default='label')
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.loss_n = args.loss_n
		self.include_id = args.include_id
		self.label_column = args.label_column
		if self.loss_n == 'BCE':
			self.loss_fn = nn.BCELoss()
	
	def loss(self, out_dict: dict):
		"""
		utilize log loss as most context-aware models
		"""
		if self.loss_n == 'BCE':
			loss = self.loss_fn(out_dict['prediction'],out_dict['label'].float())
		else:
			raise ValueError('Undefined loss function: {}'.format(self.loss_n))
		return loss
	
	class Dataset(MultiContextModel.Dataset):
		def __init__(self, model, corpus, phase):
			super().__init__(model, corpus, phase)
			self.label_column = model.label_column
			self.include_id = model.include_id

		def actions_before_epoch(self):
			pass

		def _get_feed_dict(self, index):
			user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
			label = self.data[self.label_column][index]
			feed_dict = {
				'user_id': user_id,
				'item_id': target_item,
				'label': label
			}
			feed_dict = self.get_user_features(feed_dict)
			feed_dict = self.get_item_features(feed_dict)
			feed_dict = self.get_context_features(feed_dict,index)
			
			return feed_dict

class CTRSeqModel(CTRModel):
	reader, runner = 'ContextSeqCReader', 'BaseRunner'
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--history_max', type=int, default=20,
							help='Maximum length of history.')
		return CTRModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.history_max = args.history_max
	
	class Dataset(CTRModel.Dataset):
		def __init__(self, model, corpus, phase):
			super().__init__(model, corpus, phase)
			idx_select = np.array(self.data['position']) > 0  # history length must be non-zero
			for key in self.data:
				self.data[key] = np.array(self.data[key])[idx_select]
		
		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			feed_dict = self.get_context_seq_features(feed_dict, index)
			feed_dict = self.get_item_seq_features(feed_dict)
			feed_dict['lengths'] = len(feed_dict['history_items'])

			return feed_dict

class MultiContextCFModel(MultiContextModel):
	reader, runner = 'ContextSeqCReader', 'BaseRunner'
		
	class Dataset(MultiContextModel.Dataset):
		def __init__(self, model, corpus, phase):
			super().__init__(model, corpus, phase)
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
					if (len(neg_items)==1 and neg_items[0] == -1): # exclude exploration samples
						continue
					if phase == 'train' and len(neg_items) == 0:
						continue
				if phase=='train' and self.train_exploration and self.data['c_post_repeat_vendor_c'][idx]>0: # only retain exploration interactions in training set
					continue
				if phase=='train' and self.train_repeat and self.data['c_post_repeat_vendor_c'][idx]==0: # only retain repeat interactions in training set
					continue
				if phase!='train' and self.test_exploration and self.data['c_post_repeat_vendor_c'][idx]>0: # only retain exploration interactions in test
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
			feed_dict = self.get_user_features(feed_dict)
			feed_dict = self.get_item_features(feed_dict)
			feed_dict = self.get_context_features(feed_dict,index)

			return feed_dict


class MultiContextSeqModel(MultiContextModel):
	reader, runner = 'ContextSeqCReader', 'BaseRunner'
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--history_max', type=int, default=20,
							help='Maximum length of history.')
		return MultiContextModel.parse_model_args(parser)
		
	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.history_max = args.history_max
	
	class Dataset(MultiContextModel.Dataset):
		def __init__(self, model, corpus, phase):
			super().__init__(model, corpus, phase)
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
			feed_dict = self.get_user_features(feed_dict)
			feed_dict = self.get_item_features(feed_dict)
			feed_dict = self.get_context_features(feed_dict,index)
			feed_dict = self.get_context_seq_features(feed_dict, index)
			feed_dict = self.get_item_seq_features(feed_dict)
			feed_dict['lengths'] = len(feed_dict['history_items'])

			return feed_dict

