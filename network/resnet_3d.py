# Require Pytorch Version >= 1.2.0
import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision

try:
	from . import initializer
	from .utils import load_state
except: 
	import initializer
	from utils import load_state


class RESNET18(nn.Module):

	def __init__(self, num_classes, pretrained=True, pool_first=True, **kwargs):
		super(RESNET18, self).__init__()
   
        #网络定义的地方，从库加载，更改网络类似
		self.resnet = torchvision.models.video.r3d_18(pretrained=False, progress=False, num_classes=num_classes, **kwargs)

		###################
		# Initialization #  初始化
		initializer.xavier(net=self)

		if pretrained:
			pretrained_model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/r3d_18-b3b3357e.pth') #预训练权重
			logging.info("Network:: graph initialized, loading pretrained model: `{}'".format(pretrained_model))
			assert os.path.exists(pretrained_model), "cannot locate: `{}'".format(pretrained_model)
			pretrained = torch.load(pretrained_model)  #返回一个存储，它将用作最后的反序列化
			load_state(self.resnet, pretrained)
		else:
			logging.info("Network:: graph initialized, use random inilization!")

	def forward(self, x):

		h = self.resnet(x)

		return h

#添加101		
class RESNET101(nn.Module):

	def __init__(self, num_classes, pretrained=True, pool_first=True, **kwargs):
		super(RESNET101, self).__init__()

		self.resnet = torchvision.models.resnet101(pretrained=False, progress=False, num_classes=num_classes, **kwargs)

		###################
		# Initialization #
		initializer.xavier(net=self)

		if pretrained:
			pretrained_model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/resnet101-5d3b4d8f.pth')
			logging.info("Network:: graph initialized, loading pretrained model: `{}'".format(pretrained_model))
			assert os.path.exists(pretrained_model), "cannot locate: `{}'".format(pretrained_model)
			pretrained = torch.load(pretrained_model)
			load_state(self.resnet, pretrained)
		else:
			logging.info("Network:: graph initialized, use random inilization!")

	def forward(self, x):

		h = self.resnet(x)

		return h

if __name__ == "__main__":
	logging.getLogger().setLevel(logging.DEBUG)
	# ---------
	net = RESNET18(num_classes=100, pretrained=True)
	#net = RESNET101(num_classes=100, pretrained=True)
	data = torch.autograd.Variable(torch.randn(64,3,7,7))
	output = net(data)
	print (output.shape)
