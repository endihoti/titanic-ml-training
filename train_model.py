import json
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_explore_data():
    """Load and explore the Titanic dataset"""
    print("Loading Titanic dataset...")
    df = pd.read_csv('data/titanic.csv')

    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())

    print("\nDataset info:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nSurvival rate by gender:")
    print(df.groupby('Sex')['Survived'].mean())

    print("\nSurvival rate by class:")
    print(df.groupby('Pclass')['Survived'].mean())

    return df


def preprocess_data(df):
    """Clean and preprocess the data"""
    print("\nPreprocessing data...")
    data = df.copy()

    # Handle missing values
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

    # Create new features
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

    # Extract title from name
    data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
        'Other'
    )
    data['Title'] = data['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

    # Age and Fare groups
    data['AgeGroup'] = pd.cut(
        data['Age'], bins=[0, 12, 18, 35, 60, 100],
        labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior']
    )
    data['FareGroup'] = pd.qcut(data['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])

    return data


def encode_features(data):
    """Encode categorical features"""
    print("\nEncoding categorical features...")
    features = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
        'FamilySize', 'IsAlone', 'Title', 'AgeGroup', 'FareGroup'
    ]
    model_data = data[features + ['Survived']].copy()
    label_encoders = {}
    categorical_cols = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup']

    for col in categorical_cols:
        le = LabelEncoder()
        model_data[col] = le.fit_transform(model_data[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col}: {list(le.classes_)}")

    return model_data, label_encoders


def train_model(model_data):
    """Train the Random Forest model"""
    print("\nTraining Random Forest model...")
    X = model_data.drop('Survived', axis=1)
    y = model_data['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    return model, accuracy, feature_importance, X.columns.tolist()


def save_model_artifacts(model, label_encoders, feature_names, accuracy):
    """Save model and preprocessing artifacts"""
    print("\nSaving model artifacts...")
    joblib.dump(model, 'models/titanic_model.joblib')
    joblib.dump(label_encoders, 'models/label_encoders.joblib')

    metadata = {
        'model_type': 'RandomForestClassifier',
        'training_date': datetime.now().isoformat(),
        'accuracy': float(accuracy),
        'feature_names': feature_names,
        'model_parameters': model.get_params(),
        'preprocessing_info': {
            'age_fillna': 'median',
            'embarked_fillna': 'mode',
            'fare_fillna': 'median',
            'new_features': ['FamilySize', 'IsAlone', 'Title', 'AgeGroup', 'FareGroup']
        }
    }

    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    sample_passenger = {
        'Pclass': 3, 'Sex': 'male', 'Age': 25.0, 'SibSp': 0, 'Parch': 0,
        'Fare': 8.05, 'Embarked': 'S', 'FamilySize': 1, 'IsAlone': 1,
        'Title': 'Mr', 'AgeGroup': 'Adult', 'FareGroup': 'Low'
    }
    with open('models/sample_input.json', 'w') as f:
        json.dump(sample_passenger, f, indent=2)

    print("All model artifacts saved successfully!")
    return metadata


def main():
    """Main training pipeline"""
    print("Starting Titanic Survival Prediction Model Training")
    print("=" * 60)

    df = load_and_explore_data()
    processed_data = preprocess_data(df)
    model_data, label_encoders = encode_features(processed_data)
    model, accuracy, feature_importance, feature_names = train_model(model_data)
    save_model_artifacts(model, label_encoders, feature_names, accuracy)

    print("\n" + "=" * 60)
    print(f"Training completed! Final model accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
