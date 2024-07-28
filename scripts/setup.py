import logging
import os
import pickle

import chromadb
import logfire
from custom_retriever import CustomRetriever
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# from utils import init_mongo_db

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logfire.configure()


if not os.path.exists("data/chroma-db-transformers"):
    # Download the vector database from the Hugging Face Hub if it doesn't exist locally
    # https://huggingface.co/datasets/towardsai-buster/ai-tutor-vector-db/tree/main
    logfire.warn(
        f"Vector database does not exist at 'data/chroma-db-transformers', downloading from Hugging Face Hub"
    )
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="towardsai-buster/ai-tutor-vector-db",
        local_dir="data",
        repo_type="dataset",
    )
    logfire.info(f"Downloaded vector database to 'data/chroma-db-transformers'")


def setup_database(db_collection, dict_file_name):
    db = chromadb.PersistentClient(path=f"data/{db_collection}")
    chroma_collection = db.get_or_create_collection(db_collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=OpenAIEmbedding(model="text-embedding-3-large", mode="similarity"),
        transformations=[SentenceSplitter(chunk_size=800, chunk_overlap=400)],
        show_progress=True,
        use_async=True,
    )
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
        use_async=True,
        embed_model=OpenAIEmbedding(model="text-embedding-3-large", mode="similarity"),
    )
    with open(f"data/{db_collection}/{dict_file_name}", "rb") as f:
        document_dict = pickle.load(f)

    return CustomRetriever(vector_retriever, document_dict)


# Setup retrievers
custom_retriever_transformers = setup_database(
    "chroma-db-transformers",
    "document_dict_transformers.pkl",
)
custom_retriever_peft = setup_database("chroma-db-peft", "document_dict_peft.pkl")
custom_retriever_trl = setup_database("chroma-db-trl", "document_dict_trl.pkl")
custom_retriever_llama_index = setup_database(
    "chroma-db-llama_index",
    "document_dict_llama_index.pkl",
)
custom_retriever_openai_cookbooks = setup_database(
    "chroma-db-openai_cookbooks",
    "document_dict_openai_cookbooks.pkl",
)
custom_retriever_langchain = setup_database(
    "chroma-db-langchain",
    "document_dict_langchain.pkl",
)

# Constants
CONCURRENCY_COUNT = int(os.getenv("CONCURRENCY_COUNT", 64))
MONGODB_URI = os.getenv("MONGODB_URI")

AVAILABLE_SOURCES_UI = [
    "Transformers Docs",
    "PEFT Docs",
    "TRL Docs",
    "LlamaIndex Docs",
    "OpenAI Cookbooks",
    "LangChain Docs",
    # "Towards AI Blog",
    # "RAG Course",
]

AVAILABLE_SOURCES = [
    "transformers",
    "peft",
    "trl",
    "llama_index",
    "openai_cookbooks",
    "langchain",
    # "towards_ai_blog",
    # "rag_course",
]

# mongo_db = (
#     init_mongo_db(uri=MONGODB_URI, db_name="towardsai-buster")
#     if MONGODB_URI
#     else logfire.warn("No mongodb uri found, you will not be able to save data.")
# )

__all__ = [
    "custom_retriever_transformers",
    "custom_retriever_peft",
    "custom_retriever_trl",
    "custom_retriever_llama_index",
    "custom_retriever_openai_cookbooks",
    "custom_retriever_langchain",
    "CONCURRENCY_COUNT",
    "MONGODB_URI",
    "AVAILABLE_SOURCES_UI",
    "AVAILABLE_SOURCES",
]
