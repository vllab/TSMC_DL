import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_means(means):

    k = means.shape[0]
    print('Number of means: %d' % k)
    
    nrows = math.ceil(k / 5)
    ncols = min(k, 5)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 4*nrows))
    axes = np.reshape(axes, [-1])
    for i, ax in enumerate(axes):
        if i >= k:
                break
        im = ax.imshow(np.reshape(means[i], (28,28)), vmin=0, vmax=1)
        ax.axis('off')
    plt.show()            
            