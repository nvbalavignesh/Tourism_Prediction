from huggingface_hub import HfApi
import os

repo_id = "itsjarvis/Tourism-Prediction"
repo_type = "space"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

print(f"Uploading files to existing space '{repo_id}'...")

# Upload the deployment folder to the existing space
api.upload_folder(
    folder_path="../deployment",     # the local folder containing your files
    repo_id=repo_id,          # the target repo
    repo_type=repo_type,                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)

print(f"✅ Successfully uploaded files to space '{repo_id}'!")
print(f"🚀 Your Streamlit app should be available at: https://huggingface.co/spaces/{repo_id}")
