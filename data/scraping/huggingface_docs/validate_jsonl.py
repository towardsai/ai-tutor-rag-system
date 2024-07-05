import json
from typing import Any, Dict, List


def load_and_validate_jsonl(file_path: str) -> Dict[int, Any]:
    """
    Load a .jsonl file into a dictionary and validate each line.

    Args:
    file_path (str): Path to the .jsonl file

    Returns:
    Dict[int, Any]: A dictionary where keys are line numbers (1-indexed) and values are the parsed JSON objects

    Raises:
    ValueError: If any line in the file is not valid JSON
    """
    result = {}
    with open(file_path, "r") as file:
        for line_number, line in enumerate(file, 1):
            try:
                # Strip whitespace and check if the line is empty
                stripped_line = line.strip()
                if not stripped_line:
                    print(f"Warning: Line {line_number} is empty.")
                    continue

                # Attempt to parse the JSON
                parsed_json = json.loads(stripped_line)
                result[line_number] = parsed_json
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_number}: {e}")

    return result


if __name__ == "__main__":
    file_path = "hf_transformers_v4_42_0.jsonl"
    try:
        loaded_data = load_and_validate_jsonl(file_path)
        print(f"Successfully loaded {len(loaded_data)} valid JSON objects.")

        # Optional: Print the first few items
        print("\nFirst few items:")
        for line_number, data in list(loaded_data.items())[:5]:
            print(f"Line {line_number}: {data}")

    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
