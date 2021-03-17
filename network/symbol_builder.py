import logging
import torch

from .resnet_3d import RESNET18  # This require Pytorch >= 1.2.0 support
from .resnet_3d import RESNET101

from .config import get_config

def get_symbol(name, print_net=False, **kwargs):
	
	if name.upper() == "R3D18":
		net = RESNET18(**kwargs)  #config['num_classes'] = 6  加载好网络
	elif name.upper() == "R3D101":
		net = RESNET101(**kwargs)
	else:
		logging.error("network '{}'' not implemented".format(name))
		raise NotImplementedError()

	if print_net:  #打印网络？
		logging.debug("Symbol:: Network Architecture:")
		logging.debug(net)

	input_conf = get_config(name, **kwargs)

	
	return net, input_conf

