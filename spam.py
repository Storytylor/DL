import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, SimpleRNN

# ===== 1. LOAD DATA =====
df = pd.read_csv(r"E:\DL Insem-2\spam(3).csv", encoding="latin-1")
df = df[['v1','v2']]  # keep label + message
df.columns = ['label','text']

# ===== 2. ENCODE LABEL =====
df['label'] = df['label'].map({'ham':0, 'spam':1})

# ===== 3. TOKENIZE =====
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['text'])

X = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(X, maxlen=100)

y = df['label'].values

# ===== 4. SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# ===== 5. FNN MODEL =====
# =====================================================
fnn = Sequential([
    Embedding(5000, 32, input_length=100),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

fnn.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

start = time.time()
fnn.fit(X_train, y_train, epochs=10,
        batch_size=32, validation_split=0.2)
fnn_time = time.time() - start

# =====================================================
# ===== 6. RNN MODEL =====
# =====================================================
rnn = Sequential([
    Embedding(5000, 32, input_length=100),
    SimpleRNN(64),
    Dense(1, activation='sigmoid')
])

rnn.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

start = time.time()
rnn.fit(X_train, y_train, epochs=10,
        batch_size=32, validation_split=0.2)
rnn_time = time.time() - start

# =====================================================
# ===== 7. TEST ACCURACY =====
# =====================================================
fnn_pred = (fnn.predict(X_test) > 0.5).astype(int)
rnn_pred = (rnn.predict(X_test) > 0.5).astype(int)

print("\nFNN Accuracy:",
      accuracy_score(y_test, fnn_pred))

print("RNN Accuracy:",
      accuracy_score(y_test, rnn_pred))

print("\nTraining Time:")
print("FNN:", fnn_time)
print("RNN:", rnn_time)
