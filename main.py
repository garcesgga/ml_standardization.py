import pandas as pd
from sklearn.preprocessing import StandardScaler 

data = {
    "age": [25, 32, 47, 51, 62, 23, 38, 44],
    "salary": [2500, 3200, 4700, 5100, 6200, 2300, 3800, 4400],
    "purchased": [0, 1, 1, 1, 1, 0, 0, 1]
}
df = pd.DataFrame(data)

X = df[["age", "salary"]]
y = df["purchased"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Original data:\n", X.head())
print("\nStandardized data:\n", X_scaled[:5])

