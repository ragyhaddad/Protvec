#!/bash/bin/env python 
import sys,os,json 
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns 
import numpy as np 
import calculate_properties 
import seaborn as sns 
# sns.set()
# sns.despine()
# Takes in the tsne-results file 
infile = sys.argv[1] 

df = pd.read_csv(infile,header=None)  
labels = df.iloc[:,0].values 
values = df.iloc[:,1:len(df.columns)].values 

# Set Figure
fig = plt.figure()

# Get Prop List
prop_list = calculate_properties.make_property_list(labels)  

masses = [p[0] for p in prop_list]
volume = [p[1] for p in prop_list]
vander = [p[2] for p in prop_list]
polarity = [p[3] for p in prop_list]
hydro = [p[4] for p in prop_list]
charge = [p[5] for p in prop_list]


background = 'black'
plt.rcParams['axes.facecolor'] = background
fig.patch.set_facecolor(background)
fig.suptitle('3-gram TSNE Dimensionality Reduction from 100 to 2')

alpha = 0.6
size = 2.8


cm = plt.cm.get_cmap('viridis')
plt.subplot(2, 3, 1)
plt.title('MASS')
sc = plt.scatter(values[:,0], values[:, 1],c=masses ,s=size, cmap=cm,alpha=alpha) 
plt.colorbar(sc)

plt.subplot(2, 3, 2)
plt.title('VOLUME')
sc = plt.scatter(values[:,0], values[:, 1],c=volume ,s=size, cmap=cm,alpha=alpha) 
plt.colorbar(sc)

plt.subplot(2, 3, 3)
plt.title('VAN DER WAALS')
sc = plt.scatter(values[:,0], values[:, 1],c=vander ,s=size, cmap=cm,alpha=alpha) 
plt.colorbar(sc)

plt.subplot(2, 3, 4)
plt.title('POLARITY')
sc = plt.scatter(values[:,0], values[:, 1],c=polarity ,s=size, cmap=cm,alpha=alpha) 
plt.colorbar(sc)

plt.subplot(2, 3, 5)
plt.title('HYDROPHOBICITY')
sc = plt.scatter(values[:,0], values[:, 1],c=hydro ,s=size, cmap=cm,alpha=alpha) 
plt.colorbar(sc)

plt.subplot(2, 3, 6)
plt.title('CHARGE')
sc = plt.scatter(values[:,0], values[:, 1],c=charge ,s=size, cmap=cm,alpha=alpha) 
plt.colorbar(sc)

# Show Plot
plt.show()