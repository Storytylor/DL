import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ===== 1. LOAD DATA =====
df = pd.read_csv(r"E:\DL Insem-2\diabetes(5).csv")

# ===== 2. REPLACE ZERO VALUES =====
cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for c in cols:
    df[c] = df[c].replace(0, df[c].median())

# ===== 3. FEATURES + TARGET =====
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

# ===== 4. NORMALIZE =====
X = MinMaxScaler().fit_transform(X)

# ===== 5. TRAIN TEST SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== 6. INITIAL PROTOTYPES =====
proto0 = X_train[y_train==0][0].copy()
proto1 = X_train[y_train==1][0].copy()

lr = 0.1
epochs = 100

# ===== 7. LVQ TRAINING =====
for epoch in range(epochs):
    for x, label in zip(X_train, y_train):
        d0 = np.linalg.norm(x - proto0)
        d1 = np.linalg.norm(x - proto1)

        if d0 < d1:
            winner = proto0
            win_label = 0
        else:
            winner = proto1
            win_label = 1

        if label == win_label:
            winner += lr * (x - winner)
        else:
            winner -= lr * (x - winner)

# ===== 8. TEST =====
pred = []
for x in X_test:
    d0 = np.linalg.norm(x - proto0)
    d1 = np.linalg.norm(x - proto1)
    pred.append(0 if d0 < d1 else 1)

# ===== 9. ACCURACY =====
print("Accuracy:", accuracy_score(y_test, pred))
