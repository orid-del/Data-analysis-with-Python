import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
customer_data = pd.read_csv('C:\\Users\\oridv\\OneDrive\\Documents\\bigdata\\count_article\\count_per_years.csv')
customer_data.shape
customer_data.head()
data = customer_data.iloc[:, 1:7].values
plt.figure(figsize=(10, 7))
plt.title("Words Dendrogram")
dend = shc.dendrogram(shc.linkage(data, method='ward'))
plt.show()
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)
plt.scatter(data[:,0],data[:,1],c = cluster.labels_,cmap='rainbow')
plt.xlabel('Number of articles include word in era1')
plt.ylabel('Number of Number of articles include word in era2')
plt.title('words clustering')
plt.show()