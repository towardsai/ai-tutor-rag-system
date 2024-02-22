---
title: AI Tutor Chatbot
emoji: 🧑🏻‍🏫
colorFrom: gray
colorTo: pink
sdk: gradio
app_file: scripts/gradio-ui.py
pinned: false
---
---
This project creates a helpful and accurate AI Tutor chatbot, leveraging GPT-3.5-Turbo and a RAG system. We design it to address student questions about AI with precision and clarity.

### Installation

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

### Usage

1. **Set environment variables:**

    Before running the application, you need to set up your OpenAI API key and MongoDB URI as environment variables:

    ```bash
    export OPENAI_API_KEY=your_openai_api_key_here
    export MONGODB_URI=your_mongodb_uri_here
    ```

2. **Run the application:**

    ```bash
    python scripts/gradio-ui.py
    ```

    This command starts the Gradio interface for the AI Tutor chatbot.
