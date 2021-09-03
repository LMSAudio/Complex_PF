import os
import torch
import scipy
from torch.utils import data
import numpy as np
import h5py
import scipy.io as io
import pytorch_lightning as pl

##########################
#the datasets structure is as follows: A directory contains the original dataset as .mat files, each includes the signals sig.x, sig.e, sig.s [far-end, residual, source]. This directory is scanned and the signals are stored in a .hdf5 files [to accelerate reading speed]. If the .hdf5 files are already there, the datasets will proceed without re-creating them. 
#########################

def load_data_list(folder):
    # function for reading .mat files in a folder
    directory = folder
    filelist = os.listdir(directory)
    dataList = [f for f in filelist if f.endswith(".mat")]

    print("datalist loaded...")
    return dataList

def createDataset(directory, outFile):
    # write a h5py dataset from the mat files in a directory
    # the output file name is outFile
    dataList = load_data_list(directory)
    print('extracting data from MAT files. \n')
    if not (os.path.isdir("./data/")):
        os.mkdir("./data/")

    sigLength = int(np.floor(9.9 * 16e3)) #limit all files to be of 9.9s length to avoid length mismatch

    with h5py.File("./data/" + outFile, "w", swmr=True, libver='latest') as f:
        dt = 'f'
        for fileName in dataList:
            data = io.loadmat(os.path.join(directory, fileName), struct_as_record=False)
            farend = data['sig'][0, 0].x[0:sigLength].astype(np.float32)
            residual = data['sig'][0, 0].e[0:sigLength].astype(np.float32)# data['sig'][0, 0].s[0:sigLength].astype(np.float32)+ data['sig'][0, 0].d[0:sigLength].astype(np.float32)- data['sig'][0, 0].d_est[0:sigLength].astype(np.float32)\
                #+ data['sig'][0, 0].n[0:sigLength].astype(np.float32)#data['sig'][0, 0].e[0:sigLength].astype(np.float32)
            source = data['sig'][0, 0].s[0:sigLength].astype(np.float32)

            DataSample = np.concatenate((farend, residual, source), axis=1)
            grp = f.create_group(fileName[:-4])
            grp.create_dataset('residual', data=residual)
            grp.create_dataset('source', data=source)
            grp.create_dataset('farend', data=farend)

    print('saved data. \n')

class TestDataset(data.Dataset):

    def __init__(self, matFilesPath, h5pyName):

        self.fileName = h5pyName
        if not os.path.isfile('./data/'+self.fileName):
            print('============= Creating Dataset ===================')
            createDataset(matFilesPath, self.fileName)

        with h5py.File('./data/'+self.fileName, 'r', swmr=True, libver='latest') as f:
            self.dataList = np.array(list(f.keys()))
            self.nr_samples = len(f)

        self.reader = None
        self.signalLength = 9


    def __len__(self):
        return self.nr_samples

    def __getitem__(self, idx):

        if self.reader is None:
            self.reader = h5py.File('./data/'+self.fileName, 'r', swmr=True, libver='latest')

        if torch.is_tensor(idx):
            idx = idx.tolist()

        item_name = self.dataList[idx]
        item      = self.reader[item_name]


        x = item['farend'][()].astype(np.float32)
        normfac = abs(x).max() + 1e-6
        x = x/normfac

        e = item['residual'][()].astype(np.float32)

        normfac = abs(e).max() + 1e-7
        #e = e / normfac

        signalLength = np.floor_divide(len(x), 16000)


        if signalLength<self.signalLength:
            x = np.pad(x, ((0, self.signalLength*16000 - signalLength*16000), (0, 0) ), 'constant', constant_values=(0, 0))
            e = np.pad(e, ((0, self.signalLength * 16000 - len(e[:, 0])), (0, 0)), 'constant', constant_values=(0, 0))
        batch = x, e , item_name
        try:
            s = item['source'][()].astype(np.float32)
            if signalLength < self.signalLength:
                s = np.pad(s, ((0, self.signalLength * 16000 - len(e[:, 0])), (0, 0)), 'constant',
                           constant_values=(0, 0))
            s = s[0:self.signalLength * 16000, :]
            batch = x, e, s , item_name
        except:
            None

        return batch

class TrainingDataset(data.Dataset): #This is written to output random segments out of signals. This is done to enforce truncated backpropagation through time

    def __init__(self, hop_length, winLength, input_size, memoryLength, matPath, h5pyPath):

        self.directory = h5pyPath
        if not os.path.isfile('./data/'+self.directory):
            print('============= Creating Dataset ===================')
            createDataset(matPath, self.directory)

        with h5py.File('./data/'+ self.directory, 'r', swmr=True, libver='latest') as f:
            self.DataSamples = list(f.keys())
            self.nrSamples = len(f)

        self.fs = 16000
        self.dataset = None
        self.hopSize = hop_length
        self.winLength = winLength
        self.input_size = input_size
        self.memoryLength = memoryLength
        self.signalLength = 9

        self.numberFramesPerSample = np.floor_divide(self.signalLength * self.fs, self.hopSize) # number of frames per mat file

    def __len__(self):
        return self.numberFramesPerSample*self.nrSamples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fileIdx = np.floor_divide(idx, self.numberFramesPerSample)

        frameInd = np.maximum(idx - (fileIdx)*self.numberFramesPerSample, self.input_size+1)
        SampleName = self.DataSamples[fileIdx]

        if self.dataset is None:
            self.dataset = h5py.File('./data/'+ self.directory, 'r', swmr=True, libver='latest')
        item = self.dataset[SampleName]

        x = item['farend'][()].astype(np.float32)
        e = item['residual'][()].astype(np.float32)
        s = item['source'][()].astype(np.float32)

        normfac = abs(x).max() + 1e-7

        frameInd = np.minimum(frameInd, np.floor_divide(len(x) , self.hopSize)-self.memoryLength-1)

        x = x / normfac

        farend = x[(frameInd - self.input_size+1) * self.hopSize - (
                    self.winLength - self.hopSize):(frameInd+self.memoryLength) * self.hopSize + (
                self.winLength - self.hopSize), :]

        residual = e[(frameInd - self.input_size+1) * self.hopSize - (
                    self.winLength - self.hopSize):(frameInd+self.memoryLength) * self.hopSize + (
                self.winLength - self.hopSize), :]

        source = s[(frameInd - self.input_size+1) * self.hopSize - (
                    self.winLength - self.hopSize):(frameInd+self.memoryLength) * self.hopSize + (
                self.winLength - self.hopSize), :]


        return farend, residual, source






