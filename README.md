# TSMC_DL
TSMC course materials of deep learning.

[Course slides](TSMC_DL/unsupervised-learning.pdf)

## Prerequisite
```
sudo pip3 install -r requirements.txt
```

## Mixture Model
Use BMM or GMM to fit MNIST dataset (handwritten digits), and then use the trained model to perform classification tasks.

[Demo_BMM.ipynb](Mixture_Model/Demo_BMM.ipynb) demos how to use Bernoalli mixture model.  
[Demo_GMM.ipynb](Mixture_Model/Demo_GMM.ipynb) demos how to use Gaussian mixture model.  

Please refer to [mixture.py](Mixture_Model/mixture.py), [bmm.py](Mixture_Model/bmm.py), [gmm.py](Mixture_Model/gmm.py)
for detailed implementations.

[kmeans.py](Mixture_Model/kmeans.py) perform kmeans clustering on MNIST dataset.
```
python kmeans.py --path=[Path to MNIST dataset directory. Default to "../MNIST".]
                 --k=[Number of cluster center. Default to 10.]
                 --output=[File path to save cluster centers. Default to "kmeans.dat".]
                 --verbose=[True | False. Default to False.]
```
**Note: Use python3 to run the code.**

## PCA and tSne for visualization
[Neural network for MNIST and embedding visualization](TSMC_DL/MNIST_nn_embedding.ipynb)

### How to start a tensorboard

In your code:

```python
tf.summary.FileWriter('your path to log dir', ...)
```

tensorboard command:

```
tensorboard --logdir <your path to log dir> --port <your port (defalut:6006)>
```

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
