import os
import pandas as pd
import joblib
import yaml
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from . import dispatcher  # Ensure dispatcher.py exists in the same folder with defined MODELS dict

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract config values
TRAINING_DATA = config['training_data']
TEST_DATA = config['test_data']
FOLD = config['fold']
MODEL = config['model']
FOLD_MAPPING = config['fold_mapping']

def date_transformer(df, column_name):
    """
    Extracts useful date features from a date column.
    """
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    df['is_weekend'] = df[column_name].dt.weekday >= 5
    df['day'] = df[column_name].dt.day
    df['month'] = df[column_name].dt.month
    df['year'] = df[column_name].dt.year
    return df.drop(column_name, axis=1)

if __name__ == "__main__":
    # Read data
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    df_test = df_test.loc[:, ~df_test.columns.duplicated()]

    # Apply date transformations
    df = date_transformer(df, 'Policy Start Date')
    df = date_transformer(df, 'Policy Renewal Date')

    # Split into training and validation sets
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold == FOLD]

    # Ensure the target variable matches the number of rows in Xtrain
    ytrain = train_df['Fraud_Label']
    yvalid = valid_df['Fraud_Label']

    # Drop irrelevant columns and remove 'Claim_Description' for a temporary fix
    Xtrain = train_df.drop(['Claim_ID', 'Fraud_Label', 'kfold', 'Claim_Description'], axis=1)
    Xvalid = valid_df.drop(['Claim_ID', 'Fraud_Label', 'kfold', 'Claim_Description'], axis=1)

    # Check the shapes to ensure consistency
    print(f"Shape of Xtrain: {Xtrain.shape}")
    print(f"Shape of ytrain: {ytrain.shape}")
    print(f"Shape of Xvalid: {Xvalid.shape}")
    print(f"Shape of yvalid: {yvalid.shape}")

    # Identify column types
    numerical_cols = [col for col in Xtrain.columns if Xtrain[col].dtype in ['int64', 'float64']]
    categorical_cols = [col for col in Xtrain.columns if Xtrain[col].dtype == 'object']

    # Pipelines for preprocessing
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine all pipelines for numerical and categorical columns
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    # Load model
    model = dispatcher.MODELS[MODEL]

    # Create the full pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Train the pipeline
    model_pipeline.fit(Xtrain, ytrain)

    # Validate
    ypred = model_pipeline.predict(Xvalid)
    acc = accuracy_score(yvalid, ypred)
    print(f'Accuracy: {acc:.4f}')

    # Save model and column names
    os.makedirs("../model", exist_ok=True)
    joblib.dump(model_pipeline, f"../model/{MODEL}_{FOLD}.pkl")
    joblib.dump(df.columns, f"../model/{MODEL}_{FOLD}_columns.pkl")
