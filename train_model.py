import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score

# 1. SETUP & DATA INGESTION 
# We will store all the model and data structure files in a folder named artifacts. 
ARTIFACT_DIR = "artifacts"
# if this folder named artifacts does not exist, create it. 
os.makedirs(ARTIFACT_DIR, exist_ok=True) # Create folder if missing

# Read the data for building the model. 
credit = pd.read_csv("credit.csv")
# X will contain all the predictors 
X = credit.drop(columns=["default"])
# y will contain the target variable (default) that model wants to predict using X
y = credit["default"]

# Convert categorical variables to numeric (get_dummies)
X_encoded = pd.get_dummies(X, drop_first=True)

# 2. SPLIT DATA (90% Train, 10% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.10, random_state=123
)

# 3. Create dictionary of models that we want to train. 
models = {
    "Decision Tree": DecisionTreeClassifier(criterion="entropy", random_state=123),
    "Boosting_50_3": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(criterion="entropy", max_depth=3),
        n_estimators=50,
        random_state=123
    ),
        "Boosting_50_2": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(criterion="entropy", max_depth=2),
        n_estimators=50,
        random_state=123
    ),
        "Boosting_50_5": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(criterion="entropy", max_depth=5),
        n_estimators=50,
        random_state=123
    ),
    "Random Forest_100": RandomForestClassifier(n_estimators=100, random_state=123),
    "Random Forest_50": RandomForestClassifier(n_estimators=50, random_state=123),
    "Random Forest_200": RandomForestClassifier(n_estimators=200, random_state=123),
    "Random Forest_250": RandomForestClassifier(n_estimators=250, random_state=123),
        "Boosting_100_8": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(criterion="entropy", max_depth=8),
        n_estimators=100,
        random_state=123
    )
}

# 4. We will loop through all these models to figure out which one is the best. 
# we initalize the accuracy to 0 and best_model to a null value (None keyword is null in python) 
best_acc = 0
best_model = None
best_model_name = None

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.2%}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_model_name = name

print(f"Best Model: {best_model_name} with accuracy {best_acc:.2%}")

# 5. SAVE ARTIFACTS
joblib.dump(best_model, f"{ARTIFACT_DIR}/model.pkl")
joblib.dump(list(X_train.columns), f"{ARTIFACT_DIR}/feature_columns.pkl")
