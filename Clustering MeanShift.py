#Si conocemos k, se recomienda usar kmeans o spectral clustering
#Si no conoce k, se recomienda usar meanshift, clustering jeraquico, o DBScan

#para este ejercicio se usuara un minibatchkmeans q funciona similiar al kmeans pero requiere de menor poder de computo

import pandas as pd 

from sklearn.cluster import MeanShift

dataset =  pd.read_csv('./datasets/candy.csv')
print(dataset.head())

X = dataset.drop(['competitorname'], axis=1)

meanshift = MeanShift().fit(X)

print("="*64)
print("Cuantos Cluster determino el algoritmo que se debian crear: ",len(meanshift.cluster_centers_))

#añadimos el resultado de la clasificacion al df original
print("="*64)
dataset["meanshift_cluster"]=meanshift.labels_
print(dataset)