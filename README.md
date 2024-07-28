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

This repository contains the notebooks for the RAG (Retrieval-Augmented Generation) course in the [notebooks](./notebooks) directory.

### Gradio UI Chatbot

A Gradio UI for the chatbot is available in [scripts/main.py](./scripts/main.py).

The Gradio demo is deployed on Hugging Face Spaces at: [AI Tutor Chatbot on Hugging Face](https://huggingface.co/spaces/towardsai-buster/ai-tutor-chatbot).

**Note:** A GitHub Action automatically deploys the Gradio demo when changes are pushed to the `scripts` folder.

### Installation (for Gradio UI)

1. **Create a new Python environment:**

   ```bash
   python -m venv .venv
   ```

2. **Activate the environment:**

   For macOS and Linux:

   ```bash
   source .venv/bin/activate
   ```

   For Windows:

   ```bash
   .venv\Scripts\activate
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Usage (for Gradio UI)

1. **Set environment variables:**

   Before running the application, set up the required API keys:

   For macOS and Linux:

   ```bash
   export OPENAI_API_KEY=your_openai_api_key_here
   export COHERE_API_KEY=your_cohere_api_key_here
   ```

   For Windows:

   ```bash
   set OPENAI_API_KEY=your_openai_api_key_here
   set COHERE_API_KEY=your_cohere_api_key_here
   ```

2. **Run the application:**

   ```bash
   python scripts/main.py
   ```

   This command starts the Gradio interface for the AI Tutor chatbot.
