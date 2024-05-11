from numpy import float16
from sklearn.manifold import TSNE
import torch as tr
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np


def feat_tsne_pair(x:Tensor, y:Tensor, match_mat:Tensor, name='tsne'):
    """
    x: [ns, dim] 
    y: [nt, dim]
    match_mat : {0,1} matching matrix
    name: the output png file name

    return savefig in maindir
    """
    ns = x.shape[0]
    indes = tr.where(match_mat==1)
    input = tr.cat([x, y], dim=0).numpy()
    output = tr.tensor(TSNE( n_components=2, learning_rate=100 ,init='random').fit_transform(input))
    x_embed, y_embed = output[:ns], output[ns:] # return the pair
    plt.scatter(output[:,0], output[:,1], c='b')
    for row, col in zip(*indes):
        X = [x_embed[row,0].item(),y_embed[col,0].item()]
        Y = [x_embed[row,1].item(),y_embed[col,1].item()]
        plt.scatter(X,Y)
    plt.savefig(name)
    plt.close()

    








    