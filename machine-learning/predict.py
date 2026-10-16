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


# sepal length (cm): 5.1
# sepal width (cm): 3.5
# petal length (cm): 1.4
# petal width (cm): 0.2
# Prediction: setosa