import numpy as np
from easydict import EasyDict as edict

config = edict()

config.BATCH_SIZE = 128
config.BATCH_FG_FRACTION = 0.25

config.CLS_OHEM = True
config.CLS_OHEM_RATIO = 0.7
config.BBOX_OHEM = False
config.BBOX_OHEM_RATIO = 0.7

config.EPS = 1e-14
config.LR_EPOCH = [8, 14]

# default settings
default = edict()

# default dataset
default.dataset = 'IDCard'
default.image_set = 'train'
default.root_path = 'data'
default.positive_dataset_path = 'data/IDCard'
default.negative_dataset_path = 'data/imagenet'
default.frequent = 200


default.pnet = edict()
default.pnet.train_size = 128
default.pnet.pretrained = 'model/pnet'
default.pnet.pretrained_epoch = 0
default.pnet.prefix = 'model/pnet'
default.pnet.lr = 0.01
default.pnet.begin_epoch = 0
default.pnet.end_epoch = 16

default.rnet = edict()
default.rnet.train_size = 24
default.rnet.pretrained = 'model/rnet'
default.rnet.pretrained_epoch = 0
default.rnet.prefix = 'model/rnet'
default.rnet.lr = 0.01
default.rnet.begin_epoch = 0
default.rnet.end_epoch = 16


