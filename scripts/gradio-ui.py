import os
import logging
from typing import Optional
from datetime import datetime

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import gradio as gr
from gradio.themes.utils import (
    fonts,
)

from utils import init_mongo_db

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONCURRENCY_COUNT = int(os.getenv("CONCURRENCY_COUNT", 64))
MONGODB_URI = os.getenv("MONGODB_URI")

AVAILABLE_SOURCES_UI = [
    "Gen AI 360: LLMs",
    "Gen AI 360: LangChain",
    "Gen AI 360: Advanced RAG",
    "Towards AI Blog",
    "Activeloop Docs",
    "HF Transformers Docs",
    "Wikipedia",
    "OpenAI Docs",
    "LangChain Docs",
]

AVAILABLE_SOURCES = [
    "llm_course",
    "langchain_course",
    "advanced_rag_course",
    "towards_ai",
    "activeloop",
    "hf_transformers",
    "wikipedia",
    "openai",
    "langchain_docs",
]

# Initialize MongoDB
mongo_db = (
    init_mongo_db(uri=MONGODB_URI, db_name="towardsai-buster")
    if MONGODB_URI
    else logger.warning("No mongodb uri found, you will not be able to save data.")
)

# Initialize vector store and index
db2 = chromadb.PersistentClient(path="scripts/ai-tutor-db")
chroma_collection = db2.get_or_create_collection("ai-tutor-db")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# Initialize query engine
llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0125", max_tokens=None)
embeds = OpenAIEmbedding(model="text-embedding-3-large", mode="text_search")
query_engine = index.as_query_engine(
    llm=llm, similarity_top_k=5, embed_model=embeds, streaming=True
)


AVAILABLE_SOURCES_UI = [
    "Gen AI 360: LLMs",
    "Gen AI 360: LangChain",
    "Gen AI 360: Advanced RAG",
    "Towards AI Blog",
    "Activeloop Docs",
    "HF Transformers Docs",
    "Wikipedia",
    "OpenAI Docs",
    "LangChain Docs",
]

AVAILABLE_SOURCES = [
    "llm_course",
    "langchain_course",
    "advanced_rag_course",
    "towards_ai",
    "activeloop",
    "hf_transformers",
    "wikipedia",
    "openai",
    "langchain_docs",
]


def save_completion(completion, history):
    collection = "completion_data-hf"

    # Convert completion to JSON and ignore certain columns
    completion_json = completion.to_json(
        columns_to_ignore=["embedding", "similarity", "similarity_to_answer"]
    )

    # Add the current date and time to the JSON
    completion_json["timestamp"] = datetime.utcnow().isoformat()
    completion_json["history"] = history
    completion_json["history_len"] = len(history)

    try:
        mongo_db[collection].insert_one(completion_json)
        logger.info("Completion saved to db")
    except Exception as e:
        logger.info(f"Something went wrong logging completion to db: {e}")


def log_likes(completion, like_data: gr.LikeData):
    collection = "liked_data-test"

    completion_json = completion.to_json(
        columns_to_ignore=["embedding", "similarity", "similarity_to_answer"]
    )
    completion_json["liked"] = like_data.liked
    logger.info(f"User reported {like_data.liked=}")

    try:
        mongo_db[collection].insert_one(completion_json)
        logger.info("")
    except:
        logger.info("Something went wrong logging")


def log_emails(email: gr.Textbox):
    collection = "email_data-test"

    logger.info(f"User reported {email=}")
    email_document = {"email": email}

    try:
        mongo_db[collection].insert_one(email_document)
        logger.info("")
    except:
        logger.info("Something went wrong logging")

    return ""


def format_sources(completion) -> str:
    if len(completion.source_nodes) == 0:
        return ""

    # Mapping of source system names to user-friendly names
    display_source_to_ui = {
        src: ui for src, ui in zip(AVAILABLE_SOURCES, AVAILABLE_SOURCES_UI)
    }

    documents_answer_template: str = (
        "📝 Here are the sources I used to answer your question:\n\n{documents}\n\n{footnote}"
    )
    document_template: str = (
        "[🔗 {source}: {title}]({url}), relevance: {score:2.2f}"  # Adjusted to include URL and format score as relevance
    )

    documents = "\n".join(
        [
            document_template.format(
                title=src.metadata["title"],
                score=src.score,
                source=display_source_to_ui.get(
                    src.metadata["source"], src.metadata["source"]
                ),
                url=src.metadata["url"],
            )
            for src in completion.source_nodes
        ]
    )
    footnote: str = "I'm a bot 🤖 and not always perfect."

    return documents_answer_template.format(documents=documents, footnote=footnote)


def add_sources(history, completion):

    formatted_sources = format_sources(completion)
    history.append([None, formatted_sources])

    return history


def user(user_input, history):
    """Adds user's question immediately to the chat."""
    return "", history + [[user_input, None]]


def get_answer(history, sources: Optional[list[str]] = None):
    user_input = history[-1][0]

    completion = query_engine.query(user_input)

    history[-1][1] = ""
    for token in completion.response_gen:
        history[-1][1] += token
        yield history, completion


example_questions = [
    "What is the LLama model?",
    "What is a Large Language Model?",
    "What is an embedding?",
]

theme = gr.themes.Soft()
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="blue",
        font=fonts.GoogleFont("Source Sans Pro"),
        font_mono=fonts.GoogleFont("IBM Plex Mono"),
    ),
    fill_height=True,
) as demo:
    with gr.Row():
        gr.HTML(
            "<h3><center>Towards AI 🤖: A Question-Answering Bot for anything AI-related</center></h3>"
        )

    latest_completion = gr.State()

    chatbot = gr.Chatbot(
        elem_id="chatbot", show_copy_button=True, scale=2, likeable=True
    )

    with gr.Row():
        question = gr.Textbox(
            label="What's your question?",
            placeholder="Ask a question to our AI tutor here...",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary")

    with gr.Row():
        examples = gr.Examples(
            examples=example_questions,
            inputs=question,
        )
        with gr.Row():
            email = gr.Textbox(
                label="Want to receive updates about our AI tutor?",
                placeholder="Enter your email here...",
                lines=1,
                scale=3,
            )
            submit_email = gr.Button(value="Submit", variant="secondary", scale=0)

    gr.Markdown(
        "This application uses ChatGPT to search the docs for relevant information and answer questions."
    )

    completion = gr.State()

    submit.click(user, [question, chatbot], [question, chatbot], queue=False).then(
        get_answer, inputs=[chatbot], outputs=[chatbot, completion]
    ).then(add_sources, inputs=[chatbot, completion], outputs=[chatbot])
    # .then(
    # save_completion, inputs=[completion, chatbot]
    # )

    question.submit(user, [question, chatbot], [question, chatbot], queue=False).then(
        get_answer, inputs=[chatbot], outputs=[chatbot, completion]
    ).then(add_sources, inputs=[chatbot, completion], outputs=[chatbot])
    # .then(
    #     save_completion, inputs=[completion, chatbot]
    # )

    chatbot.like(log_likes, completion)

    submit_email.click(log_emails, email, email)
    email.submit(log_emails, email, email)

demo.queue()
demo.launch(debug=True, share=False)
