import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns


class TrainModel:
    def __init__(self, data):
        self.df = data
        self.models = {
            'XGBoost': {'model': None, 'result': {}},
            'LogisticRegression': {'model': None, 'result': {}},
            'GradientBoosting': {'model': None, 'result': {}},
            'DecisionTree': {'model': None, 'result': {}}
        }
        self.split_test_train_dataset()

    def split_test_train_dataset(self):
        # One-hot encode categorical features (month and fuel category codes)
        df_model = pd.get_dummies(self.df, columns=['date_month', 'CBD_VALUE', 'EVC_VALUE', 'FDIST_VALUE', 'FVC_VALUE'], drop_first=True)
        # Separate features and target
        X = df_model.drop('is_fire', axis=1)
        y = df_model['is_fire']

        # Train-test split (80% train, 20% test), stratified by the target class
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Feature scaling for continuous features (particularly for logistic regression)
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)

        # print("Training set size:", self.X_train.shape[0], "samples")
        # print("Test set size:", self.X_test.shape[0], "samples")
        # print("Number of features after encoding:", self.X_train.shape[1])

    def train_all_models(self):
        self.train_logistic_regression()
        self.train_gradient_boosting()
        self.train_decision_tree()
        self.train_xgboost()

    def train_logistic_regression(self):
        model_name = 'LogisticRegression'
        logreg = LogisticRegression(max_iter=500, random_state=42)
        logreg.fit(self.X_train_scaled, self.y_train)
        self.models[model_name]['model'] = logreg

        y_pred_log = logreg.predict(self.X_test_scaled)
        self.evaluate_model(y_pred_log, model_name)

    def train_gradient_boosting(self):
        model_name = 'GradientBoosting'
        gb_model = GradientBoostingClassifier(random_state=42)
        gb_model.fit(self.X_train, self.y_train)
        self.models[model_name]['model'] = gb_model
        
        y_pred_gb = gb_model.predict(self.X_test)
        self.evaluate_model(y_pred_gb, model_name)

    def train_decision_tree(self):
        model_name = 'DecisionTree'
        dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt_model.fit(self.X_train, self.y_train)
        self.models[model_name]['model'] = dt_model

        y_pred_dt = dt_model.predict(self.X_test)
        self.evaluate_model(y_pred_dt, model_name)

    def train_xgboost(self):
        model_name = 'XGBoost'
        # X_train = X_train.apply(pd.to_numeric, errors='coerce')
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb_model.fit(self.X_train, self.y_train)
        self.models[model_name]['model'] = xgb_model
        
        # Predict on test data
        y_pred_xgb = xgb_model.predict(self.X_test)
        self.evaluate_model(y_pred_xgb, model_name)

    def evaluate_model(self, y_pred, model_name):
        prec = precision_score(self.y_test, y_pred)
        rec  = recall_score(self.y_test, y_pred)
        f1   = f1_score(self.y_test, y_pred)
        acc  = accuracy_score(self.y_test, y_pred)
        cm   = confusion_matrix(self.y_test, y_pred)
        auc  = roc_auc_score(self.y_test, y_pred)
        
        self.models[model_name]['result'] = {
            'prec': prec,
            'rec': rec,
            'f1': f1,
            'acc': acc,
            'cm': cm,
            'auc': auc,
            'y_pred': y_pred,
            'y_true': self.y_test
        }

        # print(f"{model_name} Metrics:\n"
        #     f"Precision = {prec:.3f}, Recall = {rec:.3f}, F1-score = {f1:.3f}, Accuracy = {acc:.3f}, ROC-AUC Score = {auc_score:.3f}")
        # print("Confusion Matrix (TN, FP, FN, TP):", cm.ravel())
        return prec, rec, f1, acc, cm, auc

    def plot_confusion_matrix(self, model_name):
        cm_log = self.models[model_name]['result']['cm']
        plt.figure(figsize=(4,3))
        sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fire','Fire'], yticklabels=['No Fire','Fire'])
        plt.title("Logistic Regression - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
        
    def plot_roc_auc(self, y_pred, model_name):
        #    calculates AUC score and prints ROC_AUC plot
        auc_score = roc_auc_score(self.y_test, y_pred)

        fpr, tpr, _ = roc_curve(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        label = f'ROC Curve (AUC = {auc_score:.2f})'
        if model_name:
            label = f'{model_name} - ' + label
        plt.plot(fpr, tpr, label=label)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve{f" for {model_name}" if model_name else ""}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()
     