import io
import json
import os
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm


class HuggingfaceParser:
    def __init__(self, html, url):
        self.soup = BeautifulSoup(html, "html.parser")
        self.url = url

    def find_sections(self):
        sections = []
        main_content = self.soup.find("article", class_="md-content__inner")
        if not main_content:
            main_content = self.soup.find(
                "div", class_="main-container"
            )  # Look for main container
        if not main_content:
            main_content = self.soup.find(
                "body"
            )  # Fallback to body if nothing else found

        if not main_content:
            print(f"Error: No main content found for {self.url}")
            return sections

        # Try to find headers
        headers = main_content.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])

        if not headers:
            # If no headers, look for other structural elements
            headers = main_content.find_all(
                ["div", "p"], class_=["docstring", "section"]
            )

        if not headers:
            print(f"Warning: No headers or sections found in {self.url}")
            # If still no headers, treat the whole content as one section
            title = self.soup.title.string if self.soup.title else "Untitled"
            sections.append(
                {
                    "name": title,
                    "url": self.url,
                    "content": main_content.get_text(strip=True),
                    "level": 1,
                }
            )
            return sections

        for i, header in enumerate(headers):
            name = header.text.strip()
            header_id = header.get("id", "")
            if header_id:
                section_url = f"{self.url}#{header_id}"
            else:
                section_url = self.url

            content = self.extract_content(
                header, headers[i + 1] if i + 1 < len(headers) else None
            )
            sections.append(
                {
                    "name": name,
                    "url": section_url,
                    "content": content,
                    "level": self.get_header_level(header),
                }
            )

        return sections

    def extract_content(self, start_tag, end_tag):
        content = []
        current = start_tag.next_sibling
        while current and current != end_tag:
            if isinstance(current, str):
                content.append(current.strip())
            elif current.name == "table":
                table_html = io.StringIO(str(current))
                content.append(
                    pd.read_html(table_html)[0].to_markdown(
                        index=False, tablefmt="github"
                    )
                )
            elif current.name not in ["script", "style"]:
                content.append(current.get_text(strip=True, separator=" "))
            current = current.next_sibling
        return "\n".join(filter(None, content))

    def get_header_level(self, tag):
        if tag.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            return int(tag.name[1])
        elif "class" in tag.attrs:
            if "docstring" in tag["class"]:
                return 1
            elif "section" in tag["class"]:
                return 2
        return 1  # Default level


def is_likely_html_file(file_path):
    excluded_extensions = {".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg"}
    return file_path.suffix == "" or file_path.suffix.lower() not in excluded_extensions


def parse_saved_html_files(html_dir, base_url):
    all_sections = []
    html_files = [
        f for f in Path(html_dir).rglob("*") if f.is_file() and is_likely_html_file(f)
    ]
    print(f"Found {len(html_files)} HTML files")

    for html_file in tqdm(html_files, desc="Parsing HTML files"):
        try:
            with open(html_file, "r", encoding="utf-8") as file:
                html_content = file.read()

            relative_path = html_file.relative_to(html_dir)
            url = urljoin(base_url, str(relative_path).replace(os.path.sep, "/"))

            parser = HuggingfaceParser(html_content, url)
            sections = parser.find_sections()

            if not sections:
                print(f"Warning: No sections found in {html_file}")
                # exit(0)
                # break
            all_sections.extend(sections)
        except Exception as e:
            print(f"Error parsing {html_file}: {str(e)}")
            # exit(0)

    return all_sections


def save_to_jsonl(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def main():
    # html_dir = "transformers_docs_v4.42.0"  # Directory where HTML files are saved
    # base_url = "https://huggingface.co/docs/transformers/"

    html_dir = "peft_docs_v0.11.0"  # Directory where HTML files are saved
    base_url = "https://huggingface.co/docs/peft/"

    output_file = "hf_peft_v0_11_0.jsonl"

    all_sections = parse_saved_html_files(html_dir, base_url)
    save_to_jsonl(all_sections, output_file)

    print(f"Parsed content saved to {output_file}")
    print(f"Total sections parsed: {len(all_sections)}")


if __name__ == "__main__":
    main()
