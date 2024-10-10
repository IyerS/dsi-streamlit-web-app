# Import required 
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score

# Import Sample Data
my_df = pd.read_csv("C:/Users/Administrator/Documents/DS-Infinity/PY ML/purchase_data.csv")
# split data into input and output variables
X = my_df.drop("purchase", axis=1)
y = my_df["purchase"]
# split the input and output variables into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Specify numeric and categorical features
numeric_features = ["age","credit_score"]
categorical_features = ["gender"]

###########################################
# Set up Pipeline
###########################################

# Numerical Feature Transformation
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer()),
                                      ("scaler", StandardScaler())])

# Categorical Feature Transformation
catagorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy = "constant", fill_value = "U")),
                                      ("ohe", OneHotEncoder(handle_unknown="ignore"))])

# Pre-processing Pipeline
preprocessing_pipeline = ColumnTransformer(transformers=[("numeric", numeric_transformer, numeric_features),
                                                         ("categorical", catagorical_transformer, categorical_features)])

###########################################
# Apply the pipeline
###########################################

# Logistic Regression
clf = Pipeline(steps=[("preprocessing_pipeline", preprocessing_pipeline),
                      ("classifier", LogisticRegression(random_state=42))])
clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)
accuracy_score(y_test, y_pred_class)

# Random Forest
clf = Pipeline(steps=[("preprocessing_pipeline", preprocessing_pipeline),
                      ("classifier", RandomForestClassifier(random_state=42))])
clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)
accuracy_score(y_test, y_pred_class)

############################################
# Save Pipeline
############################################
import joblib
joblib.dump(clf,"C:/Users/Administrator/Documents/PurchasePridiction/model.joblib")

