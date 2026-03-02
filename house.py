import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r"E:\DL Insem-2\data.csv")
df = df.head(1000)

features = ['bedrooms','bathrooms','sqft_living',
            'floors','condition','grade','yr_built']

X = df[features].dropna().values
y = df.loc[df[features].dropna().index,'price'].values.reshape(-1,1)

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42)

k = 10
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_train)
centers = kmeans.cluster_centers_

dists = []
for i in range(len(centers)):
    for j in range(i+1,len(centers)):
        dists.append(np.linalg.norm(centers[i]-centers[j]))
sigma = np.mean(dists)

def rbf(x,c,s):
    return np.exp(-np.linalg.norm(x-c)**2/(2*s**2))

def build_phi(X, centers, sigma):
    phi = np.zeros((X.shape[0],len(centers)))
    for i,x in enumerate(X):
        for j,c in enumerate(centers):
            phi[i,j] = rbf(x,c,sigma)
    return phi

phi_train = build_phi(X_train,centers,sigma)
phi_test = build_phi(X_test,centers,sigma)

W = np.linalg.lstsq(phi_train,y_train,rcond=None)[0]

y_pred = phi_test @ W
y_pred = y_scaler.inverse_transform(y_pred)
y_test_real = y_scaler.inverse_transform(y_test)

print("MSE:",mean_squared_error(y_test_real,y_pred))
print("R2 Score:",r2_score(y_test_real,y_pred))

plt.scatter(y_test_real,y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()
