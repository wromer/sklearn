import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

if __name__=="__main__":
    dt_heart = pd.read_csv('datasets/heart.csv')

    print(dt_heart.head(5))

    df_features = dt_heart.drop(['target'], axis=1)
    df_target = dt_heart['target']

    #Se se estandarizan / transformans las escalas de los features
    df_features = StandardScaler().fit_transform(df_features)
 


    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=42)

    print(X_train.shape)
    print(y_train.shape)

    #No de componentes x defecto es el min entre el num de muestras y el num de features

    pca=PCA(n_components=3)
    pca.fit(X_train)

    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

    #plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    #plt.show()

    #plt.plot(range(len(ipca.explained_variance_)), ipca.explained_variance_ratio_)
    #plt.show()

    logistic=LogisticRegression(solver='lbfgs')
    
    #Comparacion PCA vs IPCA
    #PCA Analisis de componentes principales
    df_train = pca.transform(X_train)
    df_test=pca.transform(X_test)

    logistic.fit(df_train,y_train)
    print("SCORE PCA: ", logistic.score(df_test, y_test))

    #IPCA  Incremental PCA
    df_train = ipca.transform(X_train)
    df_test=ipca.transform(X_test)

    logistic.fit(df_train,y_train)
    print("SCORE IPCA: ", logistic.score(df_test, y_test))