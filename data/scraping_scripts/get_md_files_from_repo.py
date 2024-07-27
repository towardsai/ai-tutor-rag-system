import json
import os
import random
import time

import nbformat
import requests
from nbconvert import MarkdownExporter

# GitHub repository information
owner = "huggingface"
repo = "transformers"
path = "docs/source/en"

# owner = "huggingface"
# repo = "peft"
# path = "docs/source"

# owner = "huggingface"
# repo = "trl"
# path = "docs/source"

# GitHub repository information
# owner = "run-llama"
# repo = "llama_index"
# path = "docs/docs"

# GitHub API endpoint for the repository contents
api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

# GitHub Personal Access Token (replace with your own token)
github_token = "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"

# Headers for authenticated requests
headers = {
    "Authorization": f"token {github_token}",
    "Accept": "application/vnd.github.v3+json",
}

# Maximum number of retries
MAX_RETRIES = 5


def check_rate_limit():
    rate_limit_url = "https://api.github.com/rate_limit"
    response = requests.get(rate_limit_url, headers=headers)
    data = response.json()
    remaining = data["resources"]["core"]["remaining"]
    reset_time = data["resources"]["core"]["reset"]

    if remaining < 10:  # Adjust this threshold as needed
        wait_time = reset_time - time.time()
        print(f"Rate limit nearly exceeded. Waiting for {wait_time:.2f} seconds.")
        time.sleep(wait_time + 1)  # Add 1 second buffer


def get_files_in_directory(api_url, retries=0):
    try:
        check_rate_limit()
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if retries < MAX_RETRIES:
            wait_time = (2**retries) + random.random()
            print(
                f"Error fetching directory contents: {e}. Retrying in {wait_time:.2f} seconds..."
            )
            time.sleep(wait_time)
            return get_files_in_directory(api_url, retries + 1)
        else:
            print(
                f"Failed to fetch directory contents after {MAX_RETRIES} retries: {e}"
            )
            return []


def download_file(file_url, file_path, retries=0):
    try:
        check_rate_limit()
        response = requests.get(file_url, headers=headers)
        response.raise_for_status()
        with open(file_path, "wb") as file:
            file.write(response.content)
    except requests.exceptions.RequestException as e:
        if retries < MAX_RETRIES:
            wait_time = (2**retries) + random.random()
            print(
                f"Error downloading file: {e}. Retrying in {wait_time:.2f} seconds..."
            )
            time.sleep(wait_time)
            download_file(file_url, file_path, retries + 1)
        else:
            print(f"Failed to download file after {MAX_RETRIES} retries: {e}")


def convert_ipynb_to_md(ipynb_path, md_path):
    with open(ipynb_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    exporter = MarkdownExporter()
    markdown, _ = exporter.from_notebook_node(notebook)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)


def fetch_files(api_url, local_dir):
    files = get_files_in_directory(api_url)
    for file in files:
        if file["type"] == "file" and file["name"].endswith((".md", ".mdx", ".ipynb")):
            file_url = file["download_url"]
            file_name = file["name"]
            file_path = os.path.join(local_dir, file_name)
            print(f"Downloading {file_name}...")
            download_file(file_url, file_path)

            if file_name.endswith(".ipynb"):
                md_file_name = file_name.replace(".ipynb", ".md")
                md_file_path = os.path.join(local_dir, md_file_name)
                print(f"Converting {file_name} to markdown...")
                convert_ipynb_to_md(file_path, md_file_path)
                os.remove(file_path)  # Remove the .ipynb file after conversion
        elif file["type"] == "dir":
            subdir = os.path.join(local_dir, file["name"])
            os.makedirs(subdir, exist_ok=True)
            fetch_files(file["url"], subdir)


# Local directory to save the files
local_dir = f"data/{repo}_md_files"
os.makedirs(local_dir, exist_ok=True)

# Start fetching files
fetch_files(api_url, local_dir)

print("All files have been downloaded and converted.")
