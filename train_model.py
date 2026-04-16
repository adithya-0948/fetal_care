import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
df = pd.read_csv("fetal_health.csv")

X = df.drop("fetal_health", axis=1)
y = df["fetal_health"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = ExtraTreesClassifier(n_estimators=100)
model.fit(X_scaled, y)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model trained and saved!")