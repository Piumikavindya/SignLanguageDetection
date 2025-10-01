import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../data_processing/landmarks.csv')
X = df.drop('label', axis=1).values
y = df['label'].values

le = LabelEncoder()
y_enc = le.fit_transform(y)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(clf, 'rf_landmark_model.joblib')
