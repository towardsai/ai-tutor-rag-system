import os
import logging
from typing import Optional
from datetime import datetime

import chromadb
from llama_index.core.tools import QueryEngineTool, FunctionTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.vector_stores import (
    MetadataFilters,
    MetadataFilter,
    FilterCondition,
)
import gradio as gr
from gradio.themes.utils import (
    fonts,
)

from utils import init_mongo_db
from tutor_prompts import (
    TEXT_QA_TEMPLATE,
    QueryValidation,
    system_message_validation,
    system_message_openai_agent,
)
from call_openai import api_function_call

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# This variables are used to intercept API calls
# launch mitmweb
cert_file = "/Users/omar/Downloads/mitmproxy-ca-cert.pem"
os.environ["REQUESTS_CA_BUNDLE"] = cert_file
os.environ["SSL_CERT_FILE"] = cert_file
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8080"

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

# Initialize OpenAI models
llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0125", max_tokens=None)
# embeds = OpenAIEmbedding(model="text-embedding-3-large", mode="text_search")
embeds = OpenAIEmbedding(model="text-embedding-3-large", mode="similarity")

query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=5,
    embed_model=embeds,
    streaming=True,
    text_qa_template=TEXT_QA_TEMPLATE,
)

query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="AI_information",
            description="""The 'AI_information' tool serves as a comprehensive repository for insights into the field of artificial intelligence. When utilizing this tool, the input should be the user's complete question. The input can also be adapted to focus on specific aspects or further details of the current topic under discussion. This dynamic input approach allows for a tailored exploration of AI subjects, ensuring that responses are relevant and informative. Employ this tool to fetch nuanced information on topics such as model training, fine-tuning, LLM augmentation, and more, thereby facilitating a rich, context-aware dialogue.""",
        ),
    )
]


def initialize_agent():
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=llm,
        verbose=True,
        system_prompt=system_message_openai_agent,
    )
    return agent


def reset_agent(agent_state):
    agent_state = initialize_agent()  # Reset the agent by reassigning a new instance
    return "Agent has been reset."


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
    document_template: str = "[🔗 {source}: {title}]({url}), relevance: {score:2.2f}"

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
    if completion is None:
        return history

    formatted_sources = format_sources(completion)
    history.append([None, formatted_sources])

    return history


def user(user_input, history, agent_state):
    agent = agent_state
    return "", history + [[user_input, None]]


def get_answer(history, agent_state):
    user_input = history[-1][0]
    history[-1][1] = ""

    completion = agent_state.stream_chat(user_input)

    for token in completion.response_gen:
        history[-1][1] += token
        yield history, completion

    logger.info(f"completion: {history[-1][1]=}")


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

    agent_state = gr.State(initialize_agent())

    with gr.Row():
        gr.HTML(
            "<h3><center>Towards AI 🤖: A Question-Answering Bot for anything AI-related</center></h3>"
        )

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        show_copy_button=True,
        scale=2,
        likeable=True,
        show_label=False,
    )

    with gr.Row():
        question = gr.Textbox(
            label="What's your question?",
            placeholder="Ask a question to the AI tutor here...",
            lines=1,
            scale=7,
            show_label=False,
        )
        submit = gr.Button(value="Send", variant="primary", scale=1)
        reset_button = gr.Button("Reset Chat", variant="secondary", scale=1)

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
                scale=6,
            )
            submit_email = gr.Button(value="Submit", variant="secondary", scale=1)

    gr.Markdown(
        "This application uses GPT3.5-Turbo to search the docs for relevant information and answer questions."
    )

    completion = gr.State()

    submit.click(
        user, [question, chatbot, agent_state], [question, chatbot], queue=False
    ).then(
        get_answer,
        inputs=[chatbot, agent_state],
        outputs=[chatbot, completion],
    ).then(
        add_sources, inputs=[chatbot, completion], outputs=[chatbot]
    )
    # .then(
    # save_completion, inputs=[completion, chatbot]
    # )

    question.submit(
        user, [question, chatbot, agent_state], [question, chatbot], queue=False
    ).then(
        get_answer,
        inputs=[chatbot, agent_state],
        outputs=[chatbot, completion],
    ).then(
        add_sources, inputs=[chatbot, completion], outputs=[chatbot]
    )
    # .then(
    #     save_completion, inputs=[completion, chatbot]
    # )

    reset_button.click(reset_agent, inputs=[agent_state], outputs=[agent_state])
    submit_email.click(log_emails, email, email)
    email.submit(log_emails, email, email)

demo.queue(default_concurrency_limit=CONCURRENCY_COUNT)
demo.launch(debug=False, share=False)
