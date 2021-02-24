from git_config import *


print('============= Initializing ===================')

window = torch.hann_window(n_fft).to("cuda:0")
stft = lambda x: torch.stft(x, n_fft, hop_length, window=window,  center=True)
istft = ISTFT(n_fft, hop_length, window='hanning').cuda()


trainDataset = dataset.TrainingDataset(hop_length, n_fft, inputFrameSize, memoryLength, trainingDataset_matPath, trainingDataset_h5pyname)
testDataset  = dataset.TestDataset(testDataset_matPath, testDataset_h5pyname)
valDataset   = dataset.TestDataset(valDataset_matPath, valDataset_h5pyname)


train_data_loader = DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_data_loader  = DataLoader(dataset=testDataset, batch_size=4, shuffle=True, num_workers=4)
val_data_loader   = DataLoader(dataset=valDataset, batch_size=4, shuffle=True, num_workers=4)

net = complexPF.complexPF(netcfg)

model_parameters = filter(lambda p: p.requires_grad, net.parameters())
numberOfWeights = sum([np.prod(p.size()) for p in model_parameters])
print('Number of trainable parameters: ', numberOfWeights)

print('============= Finished Initializing ===================')







