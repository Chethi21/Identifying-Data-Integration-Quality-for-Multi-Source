import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Advanced Data Cleaning
def data_cleaning(data):
    # 1.1 Handling Missing Values using KNN Imputer
    data_imputer = KNNImputer(n_neighbors=5)
    data_cleaned = pd.DataFrame(data_imputer.fit_transform(data.drop(columns=['target'])), columns=data.drop(columns=['target']).columns)
    data_cleaned['target'] = data['target'].values  # Add target column back to cleaned data
    print("Missing Values After Imputation:")
    print(data_cleaned.isnull().sum())

    # 1.2 Outlier Detection using Isolation Forest
    iso = IsolationForest(contamination=0.01, random_state=42)
    data_cleaned['Anomaly'] = iso.fit_predict(data_cleaned.drop(columns=['target']))
    data_cleaned = data_cleaned[data_cleaned['Anomaly'] == 1].drop(columns=['Anomaly'])
    print(f"Number of Outliers Removed: {len(data) - len(data_cleaned)}")

    # 1.3 Addressing Imbalanced Classes using SMOTE
    smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy='minority')
    X_resampled, y_resampled = smote.fit_resample(data_cleaned.drop(columns=['target']), data_cleaned['target'])
    print(f"Class Distribution After SMOTE: {pd.Series(y_resampled).value_counts()}")

    return X_resampled, y_resampled

# Step 2: Building and Training Models
def build_and_train_models(X_resampled, y_resampled):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # 2.1 Decision Tree Model
    decision_tree = DecisionTreeClassifier(max_depth=3)
    decision_tree.fit(X_train, y_train)
    dt_predictions = decision_tree.predict(X_test)
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_predictions):.2f}")

    # 2.2 Random Forest Model
    random_forest = RandomForestClassifier(
        random_state=42, n_estimators=30, max_depth=5, min_samples_split=10,
        min_samples_leaf=5, n_jobs=-1, max_samples=0.8, warm_start=True
    )
    random_forest.fit(X_train, y_train)
    rf_predictions = random_forest.predict(X_test)

    return y_test, dt_predictions, rf_predictions, random_forest, X_test

# Step 4: Model Evaluation
def model_evaluation(y_test, dt_predictions, rf_predictions, random_forest, X_test):
    print("\nClassification Report After Data Quality Improvements:")
    print(classification_report(y_test, rf_predictions))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, rf_predictions)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    # ROC AUC Score
    roc_auc = roc_auc_score(y_test, random_forest.predict_proba(X_test)[:, 1])
    print(f"\nROC AUC Score: {roc_auc:.2f}")

# Main Function to Run the Project
def main():
    # Sample data for demonstration (replace with actual dataset)
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, None, 6, 7],
        'feature2': [8, 9, None, 11, 12, 13, 14],
        'target': [0, 1, 1, 0, 0, 1, 0]
    })

    # Step 1: Clean the data
    X_resampled, y_resampled = data_cleaning(data)

    # Step 2: Build and train models
    y_test, dt_predictions, rf_predictions, random_forest, X_test = build_and_train_models(X_resampled, y_resampled)

    # Step 4: Evaluate the model
    model_evaluation(y_test, dt_predictions, rf_predictions, random_forest, X_test)

if __name__ == "__main__":
    main()
