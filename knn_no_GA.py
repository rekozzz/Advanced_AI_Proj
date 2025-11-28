import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load preprocessed data (optional: you can save X and y from processing.py as CSVs)
data = pd.read_csv("train.csv")

# --- Preprocessing
X = data.drop(["SalePrice", "Id"], axis=1)
y = data["SalePrice"]

# Fill missing values
X = X.fillna(0)

# Convert categorical to numeric
X = pd.get_dummies(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
y_train_class = pd.qcut(y_train, q=3, labels=["Low", "Medium", "High"])
y_test_class = pd.qcut(y_test, q=3, labels=["Low", "Medium", "High"])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train_class)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test_class, y_pred)
precision = precision_score(y_test_class, y_pred, average="macro")
recall = recall_score(y_test_class, y_pred, average="macro")
f1 = f1_score(y_test_class, y_pred, average="macro")

print("KNN Results:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")