import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler,MinMaxScaler

df=pd.read_csv(r"E:\DL Insem-2\loan_data.csv")
df.head()

df.columns
df=pd.get_dummies(df)

target="loan_status"
X=df.drop(target,axis=1)
y=df[target]

X=StandardScaler().fit_transform(X)

#early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential([
    Dense(64,activation="relu"),
    Dense(32,activation="relu"),
    Dense(1,activation="sigmoid")
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])    

from sklearn.metrics import accuracy_score
model.fit(X_train,y_train,epochs=50,batch_size=32,validation_split=.2,callbacks=[early_stop])

pred=(model.predict(X_test)>.5).astype(int)

print("Accuracy:",accuracy_score(y_test,pred))
