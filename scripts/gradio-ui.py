import json
import logging
import os
import pickle
from datetime import datetime
from typing import Optional

import chromadb
import gradio as gr
from dotenv import load_dotenv
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

load_dotenv(".env")

# from utils import init_mongo_db

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("gradio").setLevel(logging.INFO)
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
    "Towards AI Blog",
    "Wikipedia",
    "OpenAI Docs",
    "LangChain Docs",
    "LLama-Index Docs",
    "RAG Course",
]

AVAILABLE_SOURCES = [
    "HF_Transformers",
    "towards_ai_blog",
    "wikipedia",
    "openai_docs",
    "langchain_docs",
    "llama_index_docs",
    "rag_course",
]

# # Initialize MongoDB
# mongo_db = (
#     init_mongo_db(uri=MONGODB_URI, db_name="towardsai-buster")
#     if MONGODB_URI
#     else logger.warning("No mongodb uri found, you will not be able to save data.")
# )


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


with open("scripts/ai-tutor-vector-db/document_dict.pkl", "rb") as f:
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


def add_sources(answer_str, completion):
    if completion is None:
        yield answer_str

    formatted_sources = format_sources(completion)
    if formatted_sources == "":
        yield answer_str

    answer_str += "\n\n" + formatted_sources
    yield answer_str


def generate_completion(
    query,
    history,
    model,
    sources,
):

    print(f"query: {query}")
    nodes = retriever.retrieve(query)

    # Filter out nodes with the same ref_doc_id
    def filter_nodes_by_unique_doc_id(nodes):
        unique_nodes = {}
        for node in nodes:
            doc_id = node.node.ref_doc_id
            if doc_id is not None and doc_id not in unique_nodes:
                unique_nodes[doc_id] = node
        return list(unique_nodes.values())

    nodes = filter_nodes_by_unique_doc_id(nodes)
    print(f"number of nodes after filtering: {len(nodes)}")

    nodes_context = []
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
            nodes_context.append(new_node)
        else:
            nodes_context.append(node)

    if model == "gemini-1.5-flash" or model == "gemini-1.5-pro":
        llm = Gemini(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model=f"models/{model}",
            temperature=1,
            max_tokens=None,
        )
    else:
        llm = OpenAI(temperature=1, model=model, max_tokens=None)

    response_synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode="simple_summarize",
        text_qa_template=TEXT_QA_TEMPLATE,
        streaming=True,
    )

    completion = response_synthesizer.synthesize(query, nodes=nodes_context)

    answer_str = ""
    for token in completion.response_gen:
        answer_str += token
        yield answer_str

    logger.info(f"completion: {answer_str=}")

    for sources in add_sources(answer_str, completion):
        yield sources

    logger.info(f"source: {sources=}")


accordion = gr.Accordion(label="Customize Sources (Click to expand)", open=False)
model = gr.Dropdown(
    [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gpt-3.5-turbo",
    ],
    label="Model",
    value="gemini-1.5-pro",
    interactive=True,
)

sources = gr.CheckboxGroup(
    AVAILABLE_SOURCES_UI, label="Sources", value="HF Transformers", interactive=False
)


def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value["value"])
    else:
        print("You downvoted this response: " + data.value["value"])


with gr.Blocks(
    fill_height=True,
    title="Towards AI 🤖",
    analytics_enabled=True,
) as demo:
    chatbot = gr.Chatbot(
        scale=1,
        placeholder="<strong>Towards AI 🤖: A Question-Answering Bot for anything AI-related</strong><br>",
        show_label=False,
        likeable=True,
        show_copy_button=True,
    )
    chatbot.like(vote, None, None)
    gr.ChatInterface(
        fn=generate_completion,
        chatbot=chatbot,
        undo_btn=None,
        additional_inputs=[sources, model],
        additional_inputs_accordion=accordion,
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=CONCURRENCY_COUNT)
    demo.launch(debug=False, share=False)
