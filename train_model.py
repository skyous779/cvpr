import os
import logging

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from data import iterator_factory as iter_fac
from train import metric
from train.model import model
from train.lr_scheduler import MultiFactorScheduler

def train_model(sym_net, model_prefix, dataset, input_conf, clip_length=16, train_frame_interval=2,
				resume_epoch=-1, batch_size=4, save_frequency=1, lr_base=0.01, lr_factor=0.1, lr_steps=[400000, 800000],
				end_epoch=1000, distributed=False, fine_tune=False, **kwargs):

	assert torch.cuda.is_available(), "Currently, we only support CUDA version"

	# data iterator
	iter_seed = torch.initial_seed() + 100 + max(0, resume_epoch) * 100
 
    #对视频数据进行处理，返回可迭代的数据
	train_iter = iter_fac.creat(name=dataset, batch_size=batch_size, clip_length=clip_length, train_interval=train_frame_interval,
										mean=input_conf['mean'], std=input_conf['std'], seed=iter_seed)
	
	# wapper (dynamic model) net是一个类
	net = model(net=sym_net, criterion=nn.CrossEntropyLoss().cuda(),model_prefix=model_prefix, step_callback_freq=50,
				save_checkpoint_freq=save_frequency, opt_batch_size=batch_size, )
	net.net.cuda()

	# config optimization
	param_base_layers = []
	param_new_layers = []
	name_base_layers = []
	for name, param in net.net.named_parameters():
		if fine_tune:
			if ('classifier' in name) or ('fc' in name):
				param_new_layers.append(param)
			else:
				param_base_layers.append(param)
				name_base_layers.append(name)
		else:
			param_new_layers.append(param)

	if name_base_layers:
		out = "[\'" + '\', \''.join(name_base_layers) + "\']"
		logging.info("Optimizer:: >> recuding the learning rate of {} params: {}".format(len(name_base_layers),
					 out if len(out) < 300 else out[0:150] + " ... " + out[-150:]))

	net.net = torch.nn.DataParallel(net.net).cuda()  #模块级别的数据并行网络

	#优化器定义
	optimizer = torch.optim.SGD([{'params': param_base_layers, 'lr_mult': 0.2},  {'params': param_new_layers, 'lr_mult': 1.0}],
								lr=lr_base, momentum=0.9, weight_decay=0.0001, nesterov=True)

	# load params from pretrained 3d network 重点，从保存的模型中使用权重？
	if resume_epoch > 0:
		logging.info("Initializer:: resuming model from previous training")

	# resume training: model and optimizer
	if resume_epoch < 0:
		epoch_start = 0
		step_counter = 0
	else:
		net.load_checkpoint(epoch=resume_epoch, optimizer=optimizer)
		epoch_start = resume_epoch
		step_counter = epoch_start * train_iter.__len__()

	# set learning rate scheduler
	num_worker = dist.get_world_size() if torch.distributed.is_initialized() else 1
	lr_scheduler = MultiFactorScheduler(base_lr=lr_base, steps=[int(x/(batch_size*num_worker)) for x in lr_steps],
										factor=lr_factor, step_counter=step_counter)
	# define evaluation metric
	metrics = metric.MetricList(metric.Loss(name="loss-ce"), metric.Accuracy(name="top1", topk=1), metric.Accuracy(name="top5", topk=5),)

	net.fit(train_iter=train_iter, optimizer=optimizer, lr_scheduler=lr_scheduler, metrics=metrics, epoch_start=epoch_start, epoch_end=end_epoch,)
