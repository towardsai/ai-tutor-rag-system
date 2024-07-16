import json
import logging
import os
import pickle
from datetime import datetime
from typing import Optional

import chromadb
import gradio as gr
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.data_structs import Node
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, MetadataMode, NodeWithScore, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from tutor_prompts import (
    TEXT_QA_TEMPLATE,
    QueryValidation,
    system_message_openai_agent,
    system_message_validation,
)

# from utils import init_mongo_db

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# # This variables are used to intercept API calls
# # launch mitmweb
# cert_file = "/Users/omar/Downloads/mitmproxy-ca-cert.pem"
# os.environ["REQUESTS_CA_BUNDLE"] = cert_file
# os.environ["SSL_CERT_FILE"] = cert_file
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8080"

CONCURRENCY_COUNT = int(os.getenv("CONCURRENCY_COUNT", 64))
MONGODB_URI = os.getenv("MONGODB_URI")

DB_PATH = os.getenv("DB_PATH", f"scripts/ai-tutor-vector-db")
DB_COLLECTION = os.getenv("DB_NAME", "ai-tutor-vector-db")

if not os.path.exists(DB_PATH):
    # Download the vector database from the Hugging Face Hub if it doesn't exist locally
    # https://huggingface.co/datasets/towardsai-buster/ai-tutor-db/tree/main
    logger.warning(
        f"Vector database does not exist at {DB_PATH}, downloading from Hugging Face Hub"
    )
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="towardsai-buster/ai-tutor-vector-db",
        local_dir=DB_PATH,
        repo_type="dataset",
    )
    logger.info(f"Downloaded vector database to {DB_PATH}")

AVAILABLE_SOURCES_UI = [
    "HF Transformers",
]

AVAILABLE_SOURCES = [
    "HF_Transformers",
]

# # Initialize MongoDB
# mongo_db = (
#     init_mongo_db(uri=MONGODB_URI, db_name="towardsai-buster")
#     if MONGODB_URI
#     else logger.warning("No mongodb uri found, you will not be able to save data.")
# )

# Initialize vector store and index
db2 = chromadb.PersistentClient(path=DB_PATH)
chroma_collection = db2.get_or_create_collection(DB_COLLECTION)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=OpenAIEmbedding(model="text-embedding-3-large", mode="similarity"),
    transformations=[SentenceSplitter(chunk_size=800, chunk_overlap=400)],
    show_progress=True,
    use_async=True,
)

retriever = index.as_retriever(
    similarity_top_k=10,
    use_async=True,
    embed_model=OpenAIEmbedding(model="text-embedding-3-large", mode="similarity"),
)


with open("scripts/document_dict.pkl", "rb") as f:
    document_dict = pickle.load(f)


def format_sources(completion) -> str:
    if len(completion.source_nodes) == 0:
        return ""

    # Mapping of source system names to user-friendly names
    display_source_to_ui = {
        src: ui for src, ui in zip(AVAILABLE_SOURCES, AVAILABLE_SOURCES_UI)
    }

    documents_answer_template: str = (
        "📝 Here are the sources I used to answer your question:\n\n{documents}"
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

    return documents_answer_template.format(documents=documents)


def add_sources(history, completion):
    if completion is None:
        yield history

    formatted_sources = format_sources(completion)
    if formatted_sources == "":
        yield history

    history[-1][1] += "\n\n" + formatted_sources
    # history.append([None, formatted_sources])
    yield history


def user(user_input, history, agent_state):
    agent = agent_state
    return "", history + [[user_input, None]]


def get_answer(history, agent_state):
    user_input = history[-1][0]
    history[-1][1] = ""

    query = user_input

    nodes_context = []
    nodes = retriever.retrieve(query)

    # Filter nodes with the same ref_doc_id
    def filter_nodes_by_unique_doc_id(nodes):
        unique_nodes = {}
        for node in nodes:
            doc_id = node.node.ref_doc_id
            if doc_id is not None and doc_id not in unique_nodes:
                unique_nodes[doc_id] = node
        return list(unique_nodes.values())

    nodes = filter_nodes_by_unique_doc_id(nodes)
    print(len(nodes))

    for node in nodes:
        print("Node ID\t", node.node_id)
        print("Title\t", node.metadata["title"])
        print("Text\t", node.text)
        print("Score\t", node.score)
        print("Metadata\t", node.metadata)
        print("-_" * 20)
        if node.metadata["retrieve_doc"] == True:
            print("This node will be replaced by the document")
            doc = document_dict[node.node.ref_doc_id]
            print(doc.text)
            new_node = NodeWithScore(
                node=TextNode(text=doc.text, metadata=node.metadata), score=node.score
            )

            print(type(new_node))
            nodes_context.append(new_node)
        else:
            nodes_context.append(node)
            print(type(node))

    # llm = Gemini(model="models/gemini-1.5-flash", temperature=1, max_tokens=None)
    llm = Gemini(model="models/gemini-1.5-pro", temperature=1, max_tokens=None)
    # llm = OpenAI(temperature=1, model="gpt-3.5-turbo", max_tokens=None)
    # llm = OpenAI(temperature=1, model="gpt-4o", max_tokens=None)

    response_synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode="simple_summarize",
        text_qa_template=TEXT_QA_TEMPLATE,
        streaming=True,
    )

    completion = response_synthesizer.synthesize(query, nodes=nodes_context)

    for token in completion.response_gen:
        history[-1][1] += token
        yield history, completion

    logger.info(f"completion: {history[-1][1]=}")


example_questions = [
    "how to fine-tune an llm?",
    "What is a Large Language Model?",
    "What is an embedding?",
]


with gr.Blocks(fill_height=True) as demo:

    # agent_state = gr.State(initialize_agent())
    agent_state = gr.State()

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
        # reset_button = gr.Button("Reset Chat", variant="secondary", scale=1)

    # with gr.Row():
    #     examples = gr.Examples(
    #         examples=example_questions,
    #         inputs=question,
    #     )
    # with gr.Row():
    #     email = gr.Textbox(
    #         label="Want to receive updates about our AI tutor?",
    #         placeholder="Enter your email here...",
    #         lines=1,
    #         scale=6,
    #     )
    #     submit_email = gr.Button(value="Submit", variant="secondary", scale=1)

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

    # reset_button.click(
    #     reset_agent, inputs=[agent_state], outputs=[agent_state, chatbot]
    # )
    # submit_email.click(log_emails, email, email)
    # email.submit(log_emails, email, email)

demo.queue(default_concurrency_limit=CONCURRENCY_COUNT)
demo.launch(debug=False, share=False)
