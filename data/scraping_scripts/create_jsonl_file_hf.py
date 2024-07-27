import json
import os
import re
import uuid

import tiktoken

BASE_URL = "https://huggingface.co/docs/transformers/"
# BASE_URL = "https://huggingface.co/docs/peft/"
# BASE_URL = "https://huggingface.co/docs/trl/"

# List of directories to include (relative to the main input directory)
INCLUDED_DIRS = [
    # Add more directories here as needed
]

# List of directories to exclude (relative to the main input directory)
EXCLUDED_DIRS = [
    # "some_directory_to_exclude",
    # Add more directories here as needed
    "internal",
    "main_classes",
]

# List of specific files to exclude from the root directory
EXCLUDED_ROOT_FILES = [
    # "some_file_to_exclude.md",
    # Add more files here as needed
]

# Set this to True to use the INCLUDED_DIRS list, or False to use the EXCLUDED_DIRS list
USE_INCLUDE_LIST = False


def extract_title(content):
    # Try to find a Markdown title (# Title)
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if title_match:
        return title_match.group(1).strip()

    # If no Markdown title, use the first non-empty line
    lines = content.split("\n")
    for line in lines:
        if line.strip():
            return line.strip()

    # If file is empty, return None
    return None


def generate_url(file_path):
    # Remove the file extension
    path_without_extension = os.path.splitext(file_path)[0]

    # Replace backslashes with forward slashes for Windows compatibility
    path_with_forward_slashes = path_without_extension.replace("\\", "/")

    # Combine with base URL
    return BASE_URL + path_with_forward_slashes + "/"


def should_include_file(file_path):
    # Check if the file is directly in the root
    if os.path.dirname(file_path) == "":
        return os.path.basename(file_path) not in EXCLUDED_ROOT_FILES

    if USE_INCLUDE_LIST:
        # Check if the file is in one of the included directories
        return any(file_path.startswith(dir) for dir in INCLUDED_DIRS)
    else:
        # Check if the file is not in any of the excluded directories
        return not any(file_path.startswith(dir) for dir in EXCLUDED_DIRS)


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(
        encoding.encode(
            string, disallowed_special=(encoding.special_tokens_set - {"<|endoftext|>"})
        )
    )
    return num_tokens


def remove_copyright_header(content):
    # Pattern to match the copyright header
    header_pattern = re.compile(r"<!--Copyright.*?-->\s*", re.DOTALL)

    # Remove the header
    cleaned_content = header_pattern.sub("", content, count=1)

    return cleaned_content.strip()


def process_md_files(directory):
    jsonl_data = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md") or file.endswith(".mdx"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)

                # Only process the file if it should be included
                if should_include_file(relative_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    title = extract_title(content)
                    token_count = num_tokens_from_string(content, "cl100k_base")
                    if token_count < 100:
                        continue
                    cleaned_content = remove_copyright_header(content)

                    json_object = {
                        "tokens": token_count,
                        "doc_id": str(uuid.uuid4()),
                        "name": (title if title else file),
                        "url": generate_url(relative_path),
                        "retrieve_doc": (True if token_count <= 8000 else False),
                        # "source": "TRL",
                        # "source": "PEFT",
                        "source": "HF_Transformers",
                        "content": cleaned_content,
                    }

                    jsonl_data.append(json_object)

    return jsonl_data


def save_jsonl(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


# Directory where the .md files are located
input_directory = "data/transformers_md_files"
# input_directory = "data/peft_md_files"
# input_directory = "data/trl_md_files"

# Output .jsonl file
output_file = "data/transformers_data.jsonl"
# output_file = "data/peft_data.jsonl"
# output_file = "data/trl_data.jsonl"

# Process the files and save to JSONL
jsonl_data = process_md_files(input_directory)
save_jsonl(jsonl_data, output_file)

print(f"Processed {len(jsonl_data)} files and saved to {output_file}")
