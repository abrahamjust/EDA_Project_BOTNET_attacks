import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings("ignore")

# Load and sample dataset (limit to 10,000 rows for speed)
df = pd.read_csv("combined_data.csv").sample(frac=1, random_state=42).reset_index(drop=True)
df = df.sample(n=10000, random_state=42).reset_index(drop=True)

X = df.drop(columns='attack_type')
y = df['attack_type']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Evaluation function
def evaluate_model(preds, y_true):
    return {
        'Accuracy': accuracy_score(y_true, preds),
        'Precision': precision_score(y_true, preds, average='weighted', zero_division=1),
        'Recall': recall_score(y_true, preds, average='weighted', zero_division=1),
        'F1 Score': f1_score(y_true, preds, average='weighted', zero_division=1)
    }

# Models
rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
xgb = XGBClassifier(n_estimators=50, n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
ada = AdaBoostClassifier(n_estimators=50, random_state=42)
lgbm = lgb.LGBMClassifier(n_estimators=50, random_state=42)

models = {'RandomForest': rf, 'XGBoost': xgb, 'AdaBoost': ada, 'LightGBM': lgbm}

# Train & evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    scores = evaluate_model(preds, y_test)
    print(f"\n{name} Results:")
    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")

    # 3-fold Cross-validation
    cv_score = cross_val_score(model, X, y_encoded, cv=3, scoring='accuracy').mean()
    print(f"Cross-Validated Accuracy: {cv_score:.4f}")

# Stacking Ensemble
base_learners = [('rf', rf), ('xgb', xgb), ('ada', ada)]
stack_model = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression(max_iter=1000))
stack_model.fit(X_train, y_train)
stack_preds = stack_model.predict(X_test)
stack_scores = evaluate_model(stack_preds, y_test)
print(f"\nStacking Classifier Results:")
for metric, value in stack_scores.items():
    print(f"{metric}: {value:.4f}")

# Time Series Validation (if index is proxy for time)
tscv = TimeSeriesSplit(n_splits=2)  # fewer splits = faster
print("\nTimeSeriesSplit Evaluation (RandomForest):")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train)):
    rf.fit(X_train.iloc[train_idx], y_train[train_idx])
    fold_preds = rf.predict(X_train.iloc[test_idx])
    fold_scores = evaluate_model(fold_preds, y_train[test_idx])
    print(f"Fold {fold+1} - Accuracy: {fold_scores['Accuracy']:.4f}, F1 Score: {fold_scores['F1 Score']:.4f}")

# SHAP Explainability (sampled 100 rows)
print("\nGenerating SHAP Summary Plot (100 samples)...")
shap_sample = X_test.sample(100, random_state=42)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(shap_sample)
shap.summary_plot(shap_values, shap_sample, show=True)

# LIME Explainability (1 instance)
print("\nLIME Explanation for One Instance:")
lime_exp = LimeTabularExplainer(X_train.values, feature_names=X.columns.tolist(),
                                class_names=le.classes_, mode='classification')
lime_instance = lime_exp.explain_instance(X_test.iloc[0].values, rf.predict_proba)
# lime_instance.show_in_notebook()
lime_instance.save_to_file('lime_explanation.html')  # <-- Save explanation to HTML

import joblib

# Save individual models
joblib.dump(rf, 'random_forest_model1.pkl')
joblib.dump(xgb, 'xgboost_model.pkl')
joblib.dump(ada, 'adaboost_model.pkl')
joblib.dump(lgbm, 'lightgbm_model.pkl')

# Save the stacked model
joblib.dump(stack_model, 'stacking_model.pkl')

# Load the saved models
#rf_loaded = joblib.load('random_forest_model.pkl')
#xgb_loaded = joblib.load('xgboost_model.pkl')
#ada_loaded = joblib.load('adaboost_model.pkl')
#lgbm_loaded = joblib.load('lightgbm_model.pkl')

# Load the stacked model
#stack_model_loaded = joblib.load('stacking_model.pkl')

