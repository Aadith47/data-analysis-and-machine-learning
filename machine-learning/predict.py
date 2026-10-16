import joblib
import pandas as pd

model = joblib.load("iris.joblib")

values = {}
for col in model.feature_names_in_:
    values[col] = float(input(f"{col}: "))

df = pd.DataFrame([values])
names = ["setosa", "versicolor", "virginica"]

pred = model.predict(df)[0]
print("Prediction:", names[pred])