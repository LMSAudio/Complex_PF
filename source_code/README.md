
## Combining Adaptive Filtering and Complex-valued Deep Postfiltering for Acoustic Echo Cancellation 

Here you can find an implementation of our proposed complex-valued deep postfilter. 

The implementaiton builds on other source codes for *"Phase-Aware Speech Enhancement with Deep Complex U-Net", H. Choi, et al.* and *"Deep Complex Neural Networks", C. Trabelsi, et al*. These implementations can be found [here](https://github.com/chanil1218/DCUnet.pytorch) and [here](https://github.com/ChihebTrabelsi/deep_complex_networks/tree/pytorch). 

#### How to use
1. Make sure all dependencies are installed. You can find the used packages in the file [./git_config.py]
2. The provided *dummy* datasets [./data/] include: far-end, residual and near-end signals. These files are compressed and therefore, you should de-compress them before proceeding
3. To run the code using the provided *dummy* datasets, simply excute the file [./git_train.py], which should perform training, validation and testing based on the provided datasets
4. To run the algorithm on other datasets and/or using different parameterizaiton, e.g., different number of layers, you can use the file [./git_config.py] which contains most of the implementation parameters

### Implementation structure
We have used several files to separate the different parts of the implementation: 
1. git_config: includes most of the configuration options and provides easy access to many of the parameters 
2. git_init: initializes the different objects, e.g., the dataloaders and the network
3. git_train: performs the actual training, validation and testing using the specified datasets 
4. The folder [./model/] includes the classes related to the complex-valued deep postfilter model
5. The folder [./datasets/] includes the data-related classes 
6. The folder [./helpers/] includes utility functions, e.g., the loss function
### Citation
Please cite our work as

#### BibTex
>@INPROCEEDINGS{,
>
>  title={Combining Adaptive Filtering and Complex-valued Deep Postfiltering for Acoustic Echo Cancellation},
>  
>  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing},
>  
>  author={Halimeh, Mhd Modar and Haubner, Thomas and Briegleb, Annika and Schmidt, Alexander and Kellermann, Walter},
>  
>  year={2021},
>  
>  month={June},
>  
>  }

#### Plain Text
> M. M. Halimeh, T. Haubner, A. Briegleb, A. Schmidt, and W. Kellermann, "Combining Adaptive Filtering and Complex-valued Deep Postfiltering for Acoustic Echo Cancellation," 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Toronto, Canada, June, 2021.

#### RIS 
>TY  - CONF
>
>TI  - Combining Adaptive Filtering and Complex-valued Deep Postfiltering for Acoustic Echo Cancellation
>
>T2  - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
>
>AU  - M. M. Halimeh
>
>AU  - T. Haubner
>
>AU  - A. Briegleb
>
>AU  - A. Schmidt
>
>AU  - W. Kellermann
>
>PY  - 2021
>
>JO  - IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
>
>Y1  - June 2021
 
