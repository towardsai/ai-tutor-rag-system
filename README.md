# AI Tutor RAG System

Welcome to the **AI Tutor RAG (Retrieval-Augmented Generation) System** repository! This repository contains a collection of Jupyter notebooks designed to support the RAG course, focusing on techniques for enhancing AI models with retrieval-based methods.

## Course Notebooks

You can find all the course notebooks in the [notebooks](./notebooks) directory. These notebooks cover various aspects of building and fine-tuning RAG models, providing both theoretical background and practical, hands-on examples.

## Running the Notebooks

You have two options for running the code in these notebooks:

1. **Run Locally**: You can clone the repository and run the notebooks on your local machine. Install the dependencies once from [`requirements.txt`](./requirements.txt) — the notebooks' setup cells expect them to already be present outside Colab:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python -m playwright install --with-deps chromium   # crawling lesson only
   ```

   Keys are read from a `.env` file at the repository root.
2. **Run on Google Colab**: Each notebook includes a link at the top to open it directly in Google Colab, making it easy to run without local setup. The setup cell installs that notebook's pinned dependencies for you and reads keys from the Colab Secrets tab.

## About This Repository

- **Audience**: Designed for students and professionals interested in AI and natural language processing.
- **Topics Covered**: The notebooks cover foundational and advanced concepts in retrieval-augmented generation, including:
  - Data retrieval techniques
  - Model integration with retrieval systems
  - Practical applications of RAG in real-world scenarios

## Getting Started

Clone the repository and explore the notebooks at your own pace. Whether running them locally or in Colab, these notebooks will guide you step-by-step, enhancing your learning experience.
