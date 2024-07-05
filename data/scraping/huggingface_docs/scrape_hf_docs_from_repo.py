import os

import requests

# GitHub repository information
owner = "huggingface"

# repo = "peft"
# path = "docs/source"

repo = "transformers"
path = "docs/source/en"

# GitHub API endpoint for the repository contents
api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"


def get_files_in_directory(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch directory contents: {response.status_code}")
        return []


def download_file(file_url, file_path):
    response = requests.get(file_url)
    if response.status_code == 200:
        with open(file_path, "wb") as file:
            file.write(response.content)
    else:
        print(f"Failed to download file: {response.status_code}")


def fetch_md_files(api_url, local_dir):
    files = get_files_in_directory(api_url)
    for file in files:
        if file["type"] == "file" and file["name"].endswith(".md"):
            file_url = file["download_url"]
            file_path = os.path.join(local_dir, file["name"])
            print(f'Downloading {file["name"]}...')
            download_file(file_url, file_path)
        elif file["type"] == "dir":
            subdir = os.path.join(local_dir, file["name"])
            os.makedirs(subdir, exist_ok=True)
            fetch_md_files(file["url"], subdir)


# Local directory to save the files
local_dir = f"data/{repo}_docs"
os.makedirs(local_dir, exist_ok=True)

# Start fetching files
fetch_md_files(api_url, local_dir)

print("All files have been downloaded.")
