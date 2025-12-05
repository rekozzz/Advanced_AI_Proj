import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv("./data_set/data.csv")


print(data.head())


print(data.info())


if "id" in data.columns:
    data = data.drop("id", axis=1)

y = data["diagnosis"]     
X = data.drop("diagnosis", axis=1)  


y = y.map({"M": 1, "B": 0})


X = X.fillna(0)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Number of features:", X_train.shape[1])
