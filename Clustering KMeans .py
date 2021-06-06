#Si conocemos k, se recomienda usar kmeans o spectral clustering
#Si no conoce k, se recomienda usar meanshift, clustering jeraquico, o DBScan

#para este ejercicio se usuara un minibatchkmeans q funciona similiar al kmeans pero requiere de menor poder de computo

import pandas as pd 

from sklearn.cluster import MiniBatchKMeans

dataset =  pd.read_csv('./datasets/candy.csv')
print(dataset.head())

X = dataset.drop(['competitorname'], axis=1)

kmeans = MiniBatchKMeans(n_clusters=4, batch_size=10).fit(X)

print("="*64)
print("Verificamos cuantos centros o grupos tenemos:",len(kmeans.cluster_centers_))
print("="*64)
print (kmeans.predict(X))

dataset['grupo_cluster']=kmeans.predict(X)
print(dataset)