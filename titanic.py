import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ===== LOAD DATA =====
df = pd.read_csv(r"E:\DL Insem-2\train(1).csv")

# ===== PREPROCESS =====
df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

X = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].values
y = df['Survived'].values.reshape(-1,1)

X = MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== SIGMOID =====
sigmoid = lambda x: 1/(1+np.exp(-x))

# ===== WEIGHTS =====
np.random.seed(1)
W1 = np.random.randn(7,8)*0.1
W2 = np.random.randn(8,4)*0.1
W3 = np.random.randn(4,1)*0.1

b1 = np.zeros((1,8))
b2 = np.zeros((1,4))
b3 = np.zeros((1,1))

lr = 0.01

# ===== TRAIN =====
for epoch in range(1000):

    # forward
    A1 = sigmoid(X_train @ W1 + b1)
    A2 = sigmoid(A1 @ W2 + b2)
    A3 = sigmoid(A2 @ W3 + b3)

    # backprop (simple)
    d3 = A3 - y_train
    d2 = (d3 @ W3.T) * A2*(1-A2)
    d1 = (d2 @ W2.T) * A1*(1-A1)

    # update
    W3 -= lr * (A2.T @ d3)
    W2 -= lr * (A1.T @ d2)
    W1 -= lr * (X_train.T @ d1)

    b3 -= lr * np.sum(d3, axis=0)
    b2 -= lr * np.sum(d2, axis=0)
    b1 -= lr * np.sum(d1, axis=0)

    if epoch % 100 == 0:
        loss = -np.mean(
            y_train*np.log(A3+1e-8) +
            (1-y_train)*np.log(1-A3+1e-8)
        )
        print("Epoch", epoch, "Loss:", loss)

# ===== TEST =====
A1 = sigmoid(X_test @ W1 + b1)
A2 = sigmoid(A1 @ W2 + b2)
A3 = sigmoid(A2 @ W3 + b3)

pred = (A3 > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, pred))
