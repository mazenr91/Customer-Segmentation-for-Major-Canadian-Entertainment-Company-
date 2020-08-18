#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib as mpl  
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns 

from sklearn.metrics import silhouette_score, silhouette_samples
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import scipy

import pandas_profiling
from pandas_profiling import ProfileReport

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"




# In[ ]:


#loading datasets
P = pd.read_csv("file.csv")
c_d=pd.read_csv("file2.csv")
f_a=pd.read_csv("file3.csv")


# In[ ]:


#Explore P dataset

list(P)
P.shape
P.info()
P.describe().transpose()
P.head(n=20)
P.tail()
pd.isna(P)
P.corr()
sns.pairplot(P)


# In[4]:


# Explore c_d dataset
list(c_d)
c_d.shape
c_d.info()
c_d.describe().transpose()
c_d.head(n=20)
c_d.tail()
pd.isna(c_d)
c_d.corr()
sns.pairplot(c_d)


# In[5]:


# Feature Engineer E and R  

P['E']=P.P[P['P']>0]

P['E'] = P['E'].fillna(0)
P['R']=P.P[P['P']<0]

P['R'] = P['R'].fillna(0)


# In[6]:


#Aggregating P table  by pid, transaction amount sum, e sum and r sum

P=P.groupby('ID').agg(N_T=('pid','count'),ta=('TA','sum'),e=('E','sum'),r=('R','sum'))

P.info()


# In[7]:


#Feature Engineer TYPE
c_d['CODE_sliced']=c_d['CODE'].str.slice(start=1,stop=2)    
c_d['TYPE'] = np.where(c_d['CODE_sliced']=="0",'TYPE1', 'TYPE2')


# In[8]:


# Merge P and c_d datasets
P_c_d=pd.merge(c_d,P, on='ID',how="inner")


# In[9]:


#Check for correlation using pandas profile report
ProfileReport(P_c_d)


# In[10]:


#Dropping some features
P_c_d_selected=P_c_d.drop(["Var1","VAR2","CODE","CODE_sliced","VAR3", "Var4","ta","r"],axis=1)


# In[ ]:


ProfileReport(P_c_d_selected)


# In[11]:


#Normalizing P Total feature

notnormalized=P_c_d_selected['PTotal']
array=notnormalized.to_numpy()
array2=array.reshape(-1, 1) 
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_array = scaler.fit_transform(array2)
normalizedP = pd.DataFrame(data=rescaled_array, columns=["PTotal_Normalized"])
P_c_d_selected=pd.concat([P_c_d_selected, normalizedP], axis=1, ignore_index=False)
P_c_d_selected=P_c_d_selected.drop(["PTotal"],axis=1)


# In[12]:


#Normalizing N_T feature

notnormalized3=P_c_d_selected['N_T']
array3=notnormalized3.to_numpy()
array4=array3.reshape(-1, 1) 
scaler = MinMaxScaler(feature_range=(0, 1)) # function was also defined earlier
rescaled_array4 = scaler.fit_transform(array4)
normalized_NumberTransactions = pd.DataFrame(data=rescaled_array4, columns=["N_T_Normalized"])
P_c_d_selected=pd.concat([P_c_d_selected, normalized_NumberTransactions], axis=1, ignore_index=False)
P_c_d_selected=P_c_d_selected.drop(["N_T"],axis=1)


# In[13]:


#Normalizing e feature

notnormalized4=P_c_d_selected['e']
array4=notnormalized4.to_numpy()
array5=array4.reshape(-1, 1) 
scaler = MinMaxScaler(feature_range=(0, 1)) # function was also defined earlier
rescaled_array5 = scaler.fit_transform(array5)
normalized_e = pd.DataFrame(data=rescaled_array5, columns=["E_Normalized"])
P_c_d_selected=pd.concat([P_c_d_selected, normalized_e], axis=1, ignore_index=False)
P_c_d_selected=P_c_d_selected.drop(["e"],axis=1)


# In[ ]:


ProfileReport(P_c_d_selected)


# In[14]:


#Selecting features from c dataset
features=['ID','TA_tendancy','C_tendancy']
features2=f_a[features]


# In[15]:


#Creating dummy variables for features in c dataset
features2['C_tendancy_Dummy'] = np.where(features2['C_tendancy']==True,'1', '0')
features2['TA_tendancy_Dummy'] = np.where(features2['TA_tendancy']==True,'1', '0')
features3=features2.drop(['TA_tendancy','C_tendancy'],axis=1)


# In[16]:


#Merge features with combined b and P dataset
X=pd.merge(P_c_d_selected,features3, on='ID',how="inner")


# In[17]:


#Fill Nas with 0
X=X.fillna(0)
X


# In[18]:


#Dropping Unique member identifier 
scaler = StandardScaler()

X1 = X.copy()
X2=X1.drop(["ID"], axis=1)



# In[19]:


#Create dummy categorical features for TYPE and CLASS
z=pd.get_dummies(X2.TYPE, prefix='TYPE',drop_first=True)
y = pd.get_dummies(X2.CLASS, prefix='CLASS',drop_first=True)


# In[20]:


#Drop TYPE &CLASS
X2yz = pd.concat([X2, y,z], axis=1)
X3=X2yz.drop(["CLASS","TYPE"], axis=1) 
X3


# In[25]:


#Create Clusters using Agglomerative Clustering Algorithm
from sklearn.cluster import AgglomerativeClustering
np.random.seed(50)
agg = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='ward')
agg.fit(X3)
agg.labels_


# In[26]:


#Silhouette_score
silhouette_score(X3, agg.labels_)


# In[28]:


# CF (not normalized)
C_F=pd.merge(P_c_d,features3, on='ID',how="inner")


# In[42]:


#Average Transaction Feature 
C_F['Avg_Spending']=C_F.ta/C_F.N_T


# In[43]:


C_F


# In[ ]:


#Import P Type ID table 


# In[ ]:


Y=C_F


# In[ ]:


#Generate Dendogram Chart 
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy


z=hierarchy.linkage(agg.children_,'average')

import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))

dn=dendrogram(z)

plt.axhline(c='blue',linestyle='--', y=6250) 

plt.show()


# In[30]:


#View members in clusters

for label in set(agg.labels_):
    print('\nCluster{}:'.format(label))
    
    print(Y[agg.labels_==label])


# In[31]:


#Descriptive statistics for members in clusters

for label in set(agg.labels_):
    print('\nCluster{}:'.format(label))
    
    print(Y[agg.labels_==label].describe())


# In[32]:


#View members in clusters

for label in set(agg.labels_):
    print('\nCluster{}:'.format(label))
    
    print(Y[agg.labels_==label])


# In[33]:


#Defining Clusters 
cluster_0=Y[agg.labels_==0]
cluster_1=Y[agg.labels_==1]
cluster_2=Y[agg.labels_==2]
cluster_3=Y[agg.labels_==3]
cluster_4=Y[agg.labels_==4]
cluster_5=Y[agg.labels_==5]
cluster_6=Y[agg.labels_==6]


# In[34]:


#Generate Profile Report for Cluster 0
profile0 = ProfileReport(cluster_0)
profile0


# In[35]:


#Generate Profile Report for Cluster 1
profile1 = ProfileReport(cluster_1)
profile1


# In[36]:


#Generate Profile Report for Cluster 2
profile2 = ProfileReport(cluster_2)
profile2


# In[37]:


#Generate Profile Report for Cluster 3
profile3 = ProfileReport(cluster_3)
profile3


# In[38]:


#Generate Profile Report for Cluster 4
profile4 = ProfileReport(cluster_4)
profile4


# In[39]:


#Generate Profile Report for Cluster 5
profile5 = ProfileReport(cluster_5)
profile5


# In[40]:


#Generate Profile Report for Cluster 6
profile6 = ProfileReport(cluster_6)
profile6


# In[ ]:


profile0.to_file(output_file="profile0.html")
profile1.to_file(output_file="profile1.html")
profile2.to_file(output_file="profile2.html")
profile3.to_file(output_file="profile3.html")
profile4.to_file(output_file="profile4.html")
profile5.to_file(output_file="profile5.html")
profile6.to_file(output_file="profile6.html")


# In[ ]:




