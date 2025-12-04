import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


data = pd.read_csv("data.csv")



# Drop useless columns
data = data.drop(["id", "Unnamed: 32"], axis=1)

# Encode diagnosis (M = 1, B = 0)
label_encoder = LabelEncoder()
data["diagnosis"] = label_encoder.fit_transform(data["diagnosis"])
# M → 1 (Malignant), B → 0 (Benign)

# Separate features and target
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 3. Model Training
# ===============================
model = GaussianNB()
model.fit(X_train, y_train)

# ===============================
# 4. Predictions
# ===============================
y_pred = model.predict(X_test)

# ===============================
# 5. Metrics
# ===============================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


print("Naive Bayes Results on Breast Cancer Dataset:")
print("-----------------------------------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
