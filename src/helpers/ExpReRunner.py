import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List

from utils import utils
from models.BaseModel import BaseModel
from helpers.ImpressionRunner import ImpressionRunner

class ExpReRunner(ImpressionRunner):
	
	def predict(self, dataset: BaseModel.Dataset) -> np.ndarray:
		"""
		The returned prediction is a 2D-array, each row corresponds to all the candidates,
		and the ground-truth item poses the first.
		Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
				 predictions like: [[1,3,4], [2,5,6]]
		"""
		dataset.model.eval()
		dataset.model.phase = 'eval'
		predictions = list()
		exp_index = list()
		if hasattr(dataset,'phase') and dataset.phase=='dev':
			dl = DataLoader(dataset, batch_size=self.eval_batch_size_dev, shuffle=False, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		elif hasattr(dataset,'phase') and dataset.phase=='train':
			dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		else:	
			dl = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		max_len, min_len = 0,0
		for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
			if hasattr(dataset.model,'inference'):
				prediction = dataset.model.inference(utils.batch_to_gpu(batch, dataset.model.device))['prediction']
			else:
				prediction = dataset.model(utils.batch_to_gpu(batch, dataset.model.device))['prediction']
			predictions.extend(prediction.cpu().data.numpy())
			exp_index.extend((batch['repeat_time']==0).cpu().data.numpy())
			max_len, min_len = max(max_len,prediction.shape[1]), min(min_len,prediction.shape[1])
		if max_len != min_len:
			predictions = [np.pad(x,(0,max_len-len(x)),'constant',constant_values=0) for x in predictions]
		predictions = np.array(predictions)
		exp_index = np.array(exp_index)

		if dataset.model.test_all:
			rows, cols = list(), list()
			for i, u in enumerate(dataset.data['user_id']):
				clicked_items = list(dataset.corpus.train_clicked_set[u] | dataset.corpus.residual_clicked_set[u])
				idx = list(np.ones_like(clicked_items) * i)
				rows.extend(idx)
				cols.extend(clicked_items)
			predictions[rows, cols] = -np.inf

		if not self.train_models and hasattr(dataset,'phase') and dataset.phase in ['dev','test','train']:
			save_path = os.path.join(self.log_path,dataset.phase+'_prediction_%s.npy'%(self.save_appendix))
			if hasattr(dataset.model,'prediction_no_exp') or hasattr(dataset.model,'prediction_no_repeat'):
				exclude = dataset.model.prediction_no_exp if hasattr(dataset.model,'prediction_no_exp') else dataset.model.prediction_no_repeat
				save_path = os.path.join(self.log_path,dataset.phase+'_prediction_%s_exclude_other_%d.npy'%(self.save_appendix,
                                                            exclude))
			logging.info('Save %s results to %s'%(dataset.phase,save_path))
			np.save( save_path,predictions,)
		return predictions, exp_index

	
	def evaluate(self, data: BaseModel.Dataset, topks: list, metrics: list,check_sort_idx=0,all=0) -> Dict[str, float]:
		"""
		Evaluate the results for an input dataset.
		:return: result dict (key: metric@k)
		"""
		predictions, exp_index = self.predict(data)
		if data.model.test_all:
			rows, cols = list(), list()
			for i, u in enumerate(data.data['user_id']):
				clicked_items = [x[0] for x in data.corpus.user_his[u]]
				# clicked_items = [data.data['item_id'][i]]
				idx = list(np.ones_like(clicked_items) * i)
				rows.extend(idx)
				cols.extend(clicked_items)
			predictions[rows, cols] = -np.inf

		#print('data',data.data)
		rows, cols = list(), list()
		mask = np.full_like(predictions,0)
		if 'pos_num' not in data.data.keys():
			pos_num=[1 for i in range(len(predictions))]
		else:
			pos_num=data.data['pos_num']
		neg_num=data.data['neg_num']
		mp = data.model.test_max_pos_item
		mn = data.model.test_max_neg_item
		for i in range(len(data.data['neg_num'])):
			rows.extend([i for _ in range(min(pos_num[i],mp))])
			rows.extend([i for _ in range(min(neg_num[i],mn))])
			cols.extend([_ for _ in range(min(pos_num[i],mp))])
			cols.extend([_ for _ in range(mp,mp+min(neg_num[i],mn))])
		mask[rows, cols] = 1

		predictions = np.where(mask == 1,predictions,-np.inf)
		if data.phase!='train' and sum(exp_index)!=0 and sum(exp_index)<len(predictions):
			exp_results = self.evaluate_method(predictions[np.where(exp_index>0)], topks, metrics, data.model.test_all, 
									  data.data['neg_num'][np.where(exp_index>0)],mp, data.data['pos_num'][np.where(exp_index>0)],check_sort_idx,ret_all=all)
			repeat_results = self.evaluate_method(predictions[np.where(exp_index==0)], topks, metrics, data.model.test_all, 
									  data.data['neg_num'][np.where(exp_index==0)],mp, data.data['pos_num'][np.where(exp_index==0)],check_sort_idx,ret_all=all)
			exp_str = '(' + utils.format_metric(exp_results) + ')'
			repeat_str = '(' + utils.format_metric(repeat_results) + ')'
			logging.info('%s exploration results: '%(data.phase)+exp_str)
			logging.info('%s repeat results: '%(data.phase)+repeat_str)

		if 'pos_num' in data.data.keys():
			return self.evaluate_method(predictions, topks, metrics, data.model.test_all,data.data['neg_num'],mp,data.data['pos_num'],check_sort_idx,ret_all=all)
		else:
			return self.evaluate_method(predictions, topks, metrics, data.model.test_all,data.data['neg_num'],mp,check_sort_idx,ret_all=all)
