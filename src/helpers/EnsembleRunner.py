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
from sklearn.metrics import roc_auc_score

from utils import utils
from models.BaseModel import BaseModel
from helpers.ExpReRunner import ExpReRunner

class EnsembleRunner(ExpReRunner):
    
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
		intents = list()
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
				output_dict = dataset.model.inference(utils.batch_to_gpu(batch, dataset.model.device))
			else:
				output_dict = dataset.model(utils.batch_to_gpu(batch, dataset.model.device))
			prediction =output_dict['prediction']
			intent = output_dict['intents']
			predictions.extend(prediction.cpu().data.numpy())
			exp_index.extend((batch['repeat_time']==0).cpu().data.numpy())
			intents.extend(intent[:,1].cpu().data.numpy())
			max_len, min_len = max(max_len,prediction.shape[1]), min(min_len,prediction.shape[1])
		if max_len != min_len:
			predictions = [np.pad(x,(0,max_len-len(x)),'constant',constant_values=0) for x in predictions]
		predictions = np.array(predictions)
		exp_index = np.array(exp_index)
		intents = np.array(intents)

		logging.info("Intent auc: %.4f"%(roc_auc_score(exp_index.astype(float), intents)) )

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


	def train(self, data_dict: Dict[str, BaseModel.Dataset]):
		model = data_dict['dev'].model
		main_metric_results, dev_results = list(), list()
		self._check_time(start=True)
		try:
			for epoch in range(self.epoch):
				# Fit
				self._check_time()
				gc.collect()
				torch.cuda.empty_cache()
				loss = self.fit(data_dict['dev'], epoch=epoch + 1)
				if np.isnan(loss):
					logging.info("Loss is Nan. Stop training at %d."%(epoch+1))
					break
				training_time = self._check_time()

				# Observe selected tensors
				if len(model.check_list) > 0 and self.check_epoch > 0 and epoch % self.check_epoch == 0:
					utils.check(model.check_list)

				# Record dev results
				dev_result = self.evaluate(data_dict['test'], self.topk[:1], self.metrics)
				dev_results.append(dev_result)
				main_metric_results.append(dev_result[self.main_metric])
				logging_str = 'Epoch {:<5} loss={:<.4f} [{:<3.1f} s]	dev=({})'.format(
					epoch + 1, loss, training_time, utils.format_metric(dev_result))

				# Test
				if self.test_epoch > 0 and epoch % self.test_epoch  == 0:
					# train results
					train_result = self.evaluate(data_dict['train'],self.topk[:1],self.metrics, )
					logging_str += ' train=({})'.format(utils.format_metric(train_result))
					test_result = self.evaluate(data_dict['test'], self.topk[:1], self.metrics)
					logging_str += ' test=({})'.format(utils.format_metric(test_result))
				testing_time = self._check_time()
				logging_str += ' [{:<.1f} s]'.format(testing_time)

				# Save model and early stop
				if max(main_metric_results) == main_metric_results[-1] or \
						(hasattr(model, 'stage') and model.stage == 1):
					model.save_model()
					logging_str += ' *'
				logging.info(logging_str)

				if self.early_stop > 0 and self.eval_termination(main_metric_results):
					logging.info("Early stop at %d based on dev result." % (epoch + 1))
					break
		except KeyboardInterrupt:
			logging.info("Early stop manually")
			exit_here = input("Exit completely without evaluation? (y/n) (default n):")
			if exit_here.lower().startswith('y'):
				logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
				exit(1)

		# Find the best dev result across iterations
		best_epoch = main_metric_results.index(max(main_metric_results))
		logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({}) [{:<.1f} s] ".format(
			best_epoch + 1, utils.format_metric(dev_results[best_epoch]), self.time[1] - self.time[0]))
		model.load_model()
