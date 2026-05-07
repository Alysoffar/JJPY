import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load the data
df = pd.read_csv('hand_data.csv', header=None)

# 2. Separate features (coordinates) and labels (names)
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # The last column (LABEL)

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the "Brain"
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. Check how smart it is
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 6. Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)