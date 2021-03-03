import os, torch, tqdm, scipy.io.wavfile, sys
import numpy as np
from datasets import dataset
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from helpers import git_utils as utils
from model import git_complexPF as complexPF
from torch.utils.data import DataLoader
from model.istft import ISTFT
from pytorch_lightning import loggers as pl_loggers


## These are just to generate unique network names based on three flags
networkName = 'git_network'
networkType = 'complex-valued'
identifier  = 'dataType'


n_fft           = 424
hop_length      = 212
nrFrames        = 2


batch_size          = 20
nrEpochs            = 20

netcfg = {}
netcfg['encoders'] = {}
netcfg['encoders'][0] = [2, 32,  [7, 2], [2, 2], [1, 1]]
netcfg['encoders'][1] = [32, 32, [7, 2], [2, 1], [1, 1]]
netcfg['encoders'][2] = [32, 64, [7, 2], [2, 2], [1, 1]]
netcfg['encoders'][3] = [64, 32, [5, 2], [2, 1], [0, 0]]

netcfg['decoders'] = {}
netcfg['decoders'][0] = [32, 64, [5, 2], [2, 1], [1, 0]]
netcfg['decoders'][1] = [128, 64, [5, 2], [2, 2], [1, 1]]
netcfg['decoders'][2] = [96, 32, [7, 2], [2, 1], [1, 1]]
netcfg['decoders'][3] = [64,  1, [7, 2], [2, 2], [1, 1]]

netcfg['GRUdim']        = 320
netcfg['hop_length']   = hop_length
netcfg['n_fft']         = 424
netcfg['nrFrames']      = 2

memoryLength       = 20         # effectively, the length of the context window as provided to the GRU

trainingDataset_h5pyname = 'train.hdf5'
trainingDataset_matPath= 'trainingMatFilesPath'
inputFrameSize = 2

testDataset_h5pyname = 'test.hdf5'
testDataset_matPath= 'TestMatFilesPath'

valDataset_h5pyname = 'val.hdf5'
valDataset_matPath= 'ValidationMatFilesPath'
#######################################################################################################################




print('Done configuring stuff...')
