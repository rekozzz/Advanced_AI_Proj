import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("breast_cancer_selected_features.csv")
print(data.head())


print(data.info())


if "id" in data.columns:
    data = data.drop("id", axis=1)


y = data["diagnosis"]        # Target
X = data.drop("diagnosis", axis=1)  # Features


y = y.map({"M": 1, "B": 0})

X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("Training features shape:", X_train_scaled.shape)
print("Testing features shape:", X_test_scaled.shape)
print("Number of features:", X_train_scaled.shape[1])
