import pandas as pd
from sklearn.model_selection import train_test_split

# Load the training dataset
data = pd.read_csv("train.csv")

# Preview the first 5 rows
print(data.head())

# Get basic info about dataset
print(data.info())

# Count features (excluding target)
print("Number of features (excluding SalePrice):", data.shape[1] - 1)
# 1. Drop ID column (not useful for ML)
X = data.drop(["SalePrice", "Id"], axis=1)
y = data["SalePrice"]

# 2. Fill missing values
X = X.fillna(0)  # simple option; later you can do median/mode if needed

# 3. Convert categorical columns to numeric
X = pd.get_dummies(X)

# 4. Split train/test sets


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
