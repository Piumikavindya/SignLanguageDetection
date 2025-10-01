import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load CSV
df = pd.read_csv('../data_processing/landmarks.csv')
X = df.drop('label', axis=1).values
y = df['label'].values

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

# MLP Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(y_enc)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

model.save('mlp_landmarks.h5')
