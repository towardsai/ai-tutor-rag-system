"""
Hugging Face Data Upload Script

Purpose:
This script uploads a local folder to a Hugging Face dataset repository. It's designed to
update or create a dataset on the Hugging Face Hub by uploading the contents of a specified
local folder.

Usage:
- Run the script: python data/scraping_scripts/upload_dbs_to_hf.py

The script will:
- Upload the contents of the 'data' folder to the specified Hugging Face dataset repository.
- https://huggingface.co/datasets/towardsai-buster/ai-tutor-vector-db

Configuration:
- The script is set to upload to the "towardsai-buster/test-data" dataset repository. 
- It deletes all existing files in the repository before uploading (due to delete_patterns=["*"]).
"""

from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="data",
    repo_id="towardsai-buster/ai-tutor-vector-db",
    repo_type="dataset",
    multi_commits=True,
    multi_commits_verbose=True,
    delete_patterns=["*"],
    ignore_patterns=["*.jsonl", "*.py", "*.txt", "*.ipynb", "*.md", "*.pyc"],
)
