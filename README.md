# TSMC_DL
TSMC course materials of deep learning.

## PCA and tSne for visualization
[Neural network for MNIST and embedding visualization](TSMC_DL/MNIST_nn_embedding.ipynb)

In this notebook, we show how to train a neural network to do multiple-class classification in MNIST dataset.
Since the url in tesorflow example is broken, please download the dataset from [here](http://yann.lecun.com/exdb/mnist/).


Please change the data directory path to your own. 
```python
parser.add_argument('--data_dir', type=str, 
                    default='/home/tommy8054/pythonPlayground/MNIST_data/', # here!
                    help='Directory for storing input data')
```
Put your log files to desired directory.
```python
parser.add_argument('--log_dir', type=str, 
                    default='/tmp/tensorflow/mnist/logs/mnist_with_summaries', # here!
                    help='Summaries log directory')
```
Show training detail or embedding visiualization. True for training detail and False for embedding visiualization.
```python
parser.add_argument('--save_log', type=bool, default=False, # here!
                    help='Whether save log file or not')
```
## Sparse Coding
[Sparse coding using neural network](TSMC_DL/Sparse_Coding.ipynb)

In this section, we demonstrate that sparsity constrain leads to sparse feature while training. 

## Denoising Autoencoder
[Denoising Autoencoder](TSMC_DL/MNIST_Dae_Dropout.ipynb)

This is a simple example of denoising autoencoder on MNIST dataset.
