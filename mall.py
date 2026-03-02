import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom

df = pd.read_csv(r"E:\DL Insem-2\Mall_Customers(4).csv")

X = df[['Annual Income (k$)',
        'Spending Score (1-100)']].values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

som = MiniSom(x=10,y=10,input_len=2,
              sigma=1.0,learning_rate=0.5)

som.random_weights_init(X)
som.train_random(X,500)

plt.figure(figsize=(6,6))
plt.title("SOM U-Matrix")
plt.pcolor(som.distance_map(),cmap='coolwarm')
plt.colorbar()
plt.show()

cluster_count = {}
for x in X:
    w = som.winner(x)
    cluster_count[w] = cluster_count.get(w,0)+1

print("Customers per cluster:")
for k,v in cluster_count.items():
    print(k,":",v)
