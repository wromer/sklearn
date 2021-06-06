import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA

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

    kpca=KernelPCA(n_components=6, kernel='poly')
    kpca.fit(X_train)

    dt_train=kpca.transform(X_train)
    dt_test=kpca.transform(X_test)

    logistic = LogisticRegression(solver='lbfgs')
    logistic.fit(dt_train, y_train)
    print("SCORE KPA: ", logistic.score(dt_test,y_test))