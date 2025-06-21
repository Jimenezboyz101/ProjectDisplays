import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load Data
df_train = pd.read_csv('edx_train.csv')
df_test = pd.read_csv('edx_test.csv')

# Prepare Features and Labels
X = df_train.drop(columns=["userid_DI", "registered", "certified", "start_time_DI", "last_event_DI"])
y = df_train["certified"]
X_test = df_test.drop(columns=["userid_DI", "registered", "start_time_DI", "last_event_DI"], errors="ignore")

# Identify column types
numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Define Transformers
numerical_transformer = SimpleImputer(strategy="median")
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ------------------------------
# Random Forests Model
# ------------------------------
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42))
])

rf_pipeline.fit(X_train, y_train)
rf_predictions = rf_pipeline.predict(X_test)

rf_output = df_test[['userid_DI']].copy()
rf_output['certified'] = rf_predictions
rf_output.to_csv('submission_rf.csv', index=False)

# ------------------------------
# AdaBoost Model
# ------------------------------
adaboost_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", AdaBoostClassifier(
        n_estimators=300,
        learning_rate=0.25,
        random_state=42))
])

adaboost_pipeline.fit(X_train, y_train)
adaboost_predictions = adaboost_pipeline.predict(X_test)

ab_output = df_test[['userid_DI']].copy()
ab_output['certified'] = adaboost_predictions
ab_output.to_csv('submission_adaboost.csv', index=False)
