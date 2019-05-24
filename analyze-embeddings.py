#!/bin/bash/env python 
import sys,json,os 
import matplotlib.pyplot as plt 
from Bio.SeqUtils.ProtParam import ProteinAnalysis 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd 
kmers = [] 
# Parse prot file
with open(sys.argv[1],"r") as f:
    for line in f:
        line = line.strip() 
        cols = line.split(',') 
        kmer = cols[0]
        if 'X' in kmer or 'Z' in kmer or 'B' in kmer or '<' in kmer:
            continue 
        kmers.append(kmer)  



X = pd.read_csv(sys.argv[1],header=None)
print(X)

kmer_names = X.iloc[:,0].values
X = X.iloc[:, 1:len(X.columns)-1]
X = X.values 

# t-distributed Stochastic Neighbor Embedding. (Like PCA but based on similarity not covariance)
print('-- Fitting TSNE')
pca = TSNE(n_components=2)
X_trans = pca.fit_transform(X)


print("original shape:   ", X.shape)
print("transformed shape:", X_trans.shape)

plt.scatter(X_trans[:, 1], X_trans[:, 0], alpha=0.5,s=4)
df = pd.DataFrame(X_trans,index=kmer_names)
df.to_csv("3-gram-dbtx.model")
plt.show()

