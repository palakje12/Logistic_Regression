import numpy as np
import pandas as pd

import sklearn.preprocessing
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

@ignore_warnings(category=ConvergenceWarning)
def train_test(filename):
    df = pd.read_csv(filename,delimiter=',')

    type = []
    for index, row in df.iterrows():
        if row['species'] == 'Iris-setosa':
            type.append(-1)
        elif row['species'] == 'Iris-versicolor':
            type.append(0)
        elif row['species'] == 'Iris-virginica':
            type.append(1)

    df['type'] = type

    df = df.drop(df.columns[4],axis=1)

    data = df.sample(frac = 1)

    Y = data.iloc[:,-1]
    X = data.drop(data.columns[[4]],axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0)

    classifier = LogisticRegression(solver = "lbfgs", random_state= 0)
    classifier.fit(X_train, Y_train)

    predicted_y = classifier.predict(X_test)

    print(classifier.score(X_test, Y_test))

train_test(filename='IRIS.csv')