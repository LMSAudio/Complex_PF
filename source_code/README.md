
## Combining Adaptive Filtering and Complex-valued Deep Postfiltering for Acoustic Echo Cancellation 

Here you can find an implementation of our proposed complex-valued deep postfilter. 

The implementaiton builds on other implementations of *Phase-Aware Speech Enhancement with Deep Complex U-Net, H. Choi, et al.* and *Deep Complex Neural Networks, C. Trabelsi*. These implementations can be found [here][link: https://github.com/chanil1218/DCUnet.pytorch] and [here][link: https://github.com/ChihebTrabelsi/deep_complex_networks/tree/pytorch]. 

#### How to use: 
1. Make sure all dependencies are installed. You can find the used packages in the file [./git_config.py]
2. The provided *dummy* datasets [./data/] are compressed and therefore, you should de-compress the files before proceeding. 
3. To run the code using the provided *dummy* datasets, simply run the file [./git_train.py], which should perform training, validation and testing based on the provided datasets
4. To run the algorithm on other datasets and/or using different parameterizaiton, e.g., different number of layers, you can use the file [./git_config.py] which contains most of the network's parameters

The implementation uses several files to separate the different parts of the code: 
1. 


