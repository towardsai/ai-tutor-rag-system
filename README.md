---
title: AI Tutor Chatbot
emoji: 🧑🏻‍🏫
colorFrom: gray
colorTo: pink
sdk: gradio
sdk_version: 4.31.3
app_file: scripts/gradio-ui.py
pinned: false
---

This repository contains the notebooks for the RAG AI tutor course in [notebooks](./notebooks).

A Gradio UI for an AI tutor chatbot is available in [scripts](./scripts/gradio-ui.py).

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
