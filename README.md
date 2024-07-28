---
title: AI Tutor Chatbot
emoji: 🧑🏻‍🏫
colorFrom: gray
colorTo: pink
sdk: gradio
sdk_version: 4.39.0
app_file: scripts/main.py
pinned: false
---

This repository contains the notebooks for the RAG course in [notebooks](./notebooks).

A Gradio UI for an AI tutor chatbot is available in [scripts/gradio-ui.py](./scripts/gradio-ui.py).

The Gradio demo is deployed on Hugging Face Spaces service at this URL: [AI Tutor Chatbot on Hugging Face](https://huggingface.co/spaces/towardsai-buster/ai-tutor-chatbot).

There is a GitHub action that automatically deploys the Gradio demo after pushing changes inside the scripts folder.

### Installation (for Gradio UI)

1. **Create a new Python environment:**

    ```bash
    python -m venv .venv
    ```

    This command creates a virtual environment named `.venv`.

2. **Activate the environment:**

    For macOS and Linux:

    ```bash
    source .venv/bin/activate
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Usage (for Gradio UI)

1. **Set environment variables:**

    Before running the application, you need to set up your OpenAI API key:

    ```bash
    export OPENAI_API_KEY=your_openai_api_key_here
    ```

2. **Run the application:**

    ```bash
    python scripts/gradio-ui.py
    ```

    This command starts the Gradio interface for the AI Tutor chatbot.
