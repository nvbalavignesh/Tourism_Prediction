# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/itsjarvis/Tourism-Prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unnecessary columns if any (like CustomerID which is just an identifier)
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)

# Feature Engineering:
# 1. Convert 'Fe Male' to 'Female' in Gender column
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

# 2. Convert 'Unmarried' to 'Single' in MaritalStatus column
df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')

# Encode categorical features
categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'MaritalStatus',
    'ProductPitched', 'Designation'
]

# Apply label encoding to all categorical features
label_encoder = LabelEncoder()
for feature in categorical_features:
    if feature in df.columns:
        df[feature] = label_encoder.fit_transform(df[feature])

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  
        repo_id="itsjarvis/Tourism-Prediction",
        repo_type="dataset",
    )
