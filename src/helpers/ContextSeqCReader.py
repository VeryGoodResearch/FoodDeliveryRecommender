# -*- coding: UTF-8 -*-

import logging
import numpy as np
import pandas as pd
import os
import sys
from utils import utils
from helpers.ContextSeqReader import ContextSeqReader

class ContextSeqCReader(ContextSeqReader):
	"""
	Get user history with context 
	"""
	def _read_data(self):
		logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
		self.data_df = dict()
		for key in ['train', 'dev', 'test']:
			self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep).reset_index(drop=True).sort_values(by = ['user_id','time'])
			self.data_df[key] = utils.eval_list_columns(self.data_df[key])

		logging.info('Counting dataset statistics...')
		self.all_df = pd.concat([self.data_df[key][['user_id', 'item_id', 'time']] for key in ['train', 'dev', 'test']])
		self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1
		for key in ['dev', 'test']:
			if 'neg_items' in self.data_df[key]:
				neg_items = set()
				for items in self.data_df[key]['neg_items'].tolist():
					neg_items = neg_items | set(items)
				neg_items = np.array(list(neg_items))
				assert (neg_items >= self.n_items).sum() == 0  # assert negative items don't include unseen ones
		logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(
			self.n_users - 1, self.n_items - 1, len(self.all_df)))
	
	def _append_his_info(self):
		"""
		self.user_his: store user history sequence [(i1,t1), (i1,t2), ...]
		add the 'position' of each interaction in user_his to data_df
		"""
		logging.info('Appending history info...')
		data_dfs = dict()
		for key in ['train','dev','test']:
			data_dfs[key] = self.data_df[key].copy()
			data_dfs[key]['phase'] = key
		sort_df = pd.concat([data_dfs[phase][['user_id','item_id','time','phase']+self.context_feature_names] 
					   for phase in ['train','dev','test']]).sort_values(by=['time', 'user_id'], kind='mergesort')
		position = list()
		self.user_his = dict()  # store the already seen sequence of each user
		context_features = sort_df[self.context_feature_names].to_numpy()
		for idx, (uid, iid, t) in enumerate(zip(sort_df['user_id'], sort_df['item_id'], sort_df['time'])):
			if uid not in self.user_his:
				self.user_his[uid] = list()
			position.append(len(self.user_his[uid]))
			self.user_his[uid].append((iid, t, context_features[idx]))
		sort_df['position'] = position
		for key in ['train', 'dev', 'test']:
			self.data_df[key] = pd.merge(
				left=self.data_df[key], right=sort_df.drop(columns=['phase']+self.context_feature_names),
				how='left', on=['user_id', 'item_id', 'time'])
		del sort_df