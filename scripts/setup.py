import asyncio
import json
import logging
import os
import pickle

import chromadb
import logfire
from custom_retriever import CustomRetriever
from dotenv import load_dotenv
from llama_index.core import Document, SimpleKeywordTableIndex, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import (
    KeywordTableSimpleRetriever,
    VectorIndexRetriever,
)
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from utils import init_mongo_db

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logfire.configure()


if not os.path.exists("data/chroma-db-all_sources"):
    # Download the vector database from the Hugging Face Hub if it doesn't exist locally
    # https://huggingface.co/datasets/towardsai-buster/ai-tutor-vector-db/tree/main
    logfire.warn(
        f"Vector database does not exist at 'data/chroma-db-all_sources', downloading from Hugging Face Hub"
    )
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="towardsai-buster/ai-tutor-vector-db",
        local_dir="data",
        repo_type="dataset",
    )
    logfire.info(f"Downloaded vector database to 'data/chroma-db-all_sources'")


def create_docs(input_file: str) -> list[Document]:
    with open(input_file, "r") as f:
        documents = []
        for line in f:
            data = json.loads(line)
            documents.append(
                Document(
                    doc_id=data["doc_id"],
                    text=data["content"],
                    metadata={  # type: ignore
                        "url": data["url"],
                        "title": data["name"],
                        "tokens": data["tokens"],
                        "retrieve_doc": data["retrieve_doc"],
                        "source": data["source"],
                    },
                    excluded_llm_metadata_keys=[
                        "title",
                        "tokens",
                        "retrieve_doc",
                        "source",
                    ],
                    excluded_embed_metadata_keys=[
                        "url",
                        "tokens",
                        "retrieve_doc",
                        "source",
                    ],
                )
            )
    return documents


def setup_database(db_collection, dict_file_name):
    db = chromadb.PersistentClient(path=f"data/{db_collection}")
    chroma_collection = db.get_or_create_collection(db_collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    embed_model = CohereEmbedding(
        api_key=os.environ["COHERE_API_KEY"],
        model_name="embed-english-v3.0",
        input_type="search_query",
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        transformations=[SentenceSplitter(chunk_size=800, chunk_overlap=0)],
        show_progress=True,
        use_async=True,
    )
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=15,
        embed_model=embed_model,
        use_async=True,
    )
    with open(f"data/{db_collection}/{dict_file_name}", "rb") as f:
        document_dict = pickle.load(f)

    with open("data/keyword_retriever_sync.pkl", "rb") as f:
        keyword_retriever: KeywordTableSimpleRetriever = pickle.load(f)

    # # Creating the keyword index and retriever
    # logfire.info("Creating nodes from documents")
    # documents = create_docs("data/all_sources_data.jsonl")
    # pipeline = IngestionPipeline(
    #     transformations=[SentenceSplitter(chunk_size=800, chunk_overlap=0)]
    # )
    # all_nodes = pipeline.run(documents=documents, show_progress=True)
    # # with open("data/all_nodes.pkl", "wb") as f:
    # #     pickle.dump(all_nodes, f)

    # # all_nodes = pickle.load(open("data/all_nodes.pkl", "rb"))
    # logfire.info(f"Number of nodes: {len(all_nodes)}")

    # keyword_index = SimpleKeywordTableIndex(
    #     nodes=all_nodes, max_keywords_per_chunk=10, show_progress=True, use_async=False
    # )
    # # with open("data/keyword_index.pkl", "wb") as f:
    # # pickle.dump(keyword_index, f)

    # # keyword_index = pickle.load(open("data/keyword_index.pkl", "rb"))

    # logfire.info("Creating keyword retriever")
    # keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)

    # with open("data/keyword_retriever_sync.pkl", "wb") as f:
    #     pickle.dump(keyword_retriever, f)

    return CustomRetriever(vector_retriever, document_dict, keyword_retriever, "OR")


# Setup retrievers
# custom_retriever_transformers: CustomRetriever = setup_database(
#     "chroma-db-transformers",
#     "document_dict_transformers.pkl",
# )
# custom_retriever_peft: CustomRetriever = setup_database(
#     "chroma-db-peft", "document_dict_peft.pkl"
# )
# custom_retriever_trl: CustomRetriever = setup_database(
#     "chroma-db-trl", "document_dict_trl.pkl"
# )
# custom_retriever_llama_index: CustomRetriever = setup_database(
#     "chroma-db-llama_index",
#     "document_dict_llama_index.pkl",
# )
# custom_retriever_openai_cookbooks: CustomRetriever = setup_database(
#     "chroma-db-openai_cookbooks",
#     "document_dict_openai_cookbooks.pkl",
# )
# custom_retriever_langchain: CustomRetriever = setup_database(
#     "chroma-db-langchain",
#     "document_dict_langchain.pkl",
# )

custom_retriever_all_sources: CustomRetriever = setup_database(
    "chroma-db-all_sources",
    "document_dict_all_sources.pkl",
)

# Constants
CONCURRENCY_COUNT = int(os.getenv("CONCURRENCY_COUNT", 64))
MONGODB_URI = os.getenv("MONGODB_URI")

AVAILABLE_SOURCES_UI = [
    "Transformers Docs",
    "PEFT Docs",
    "TRL Docs",
    "LlamaIndex Docs",
    "LangChain Docs",
    "OpenAI Cookbooks",
    "Towards AI Blog",
    # "All Sources",
    # "RAG Course",
]

AVAILABLE_SOURCES = [
    "transformers",
    "peft",
    "trl",
    "llama_index",
    "langchain",
    "openai_cookbooks",
    "tai_blog",
    # "all_sources",
    # "rag_course",
]

mongo_db = (
    init_mongo_db(uri=MONGODB_URI, db_name="towardsai-buster")
    if MONGODB_URI
    else logfire.warn("No mongodb uri found, you will not be able to save data.")
)

__all__ = [
    # "custom_retriever_transformers",
    # "custom_retriever_peft",
    # "custom_retriever_trl",
    # "custom_retriever_llama_index",
    # "custom_retriever_openai_cookbooks",
    # "custom_retriever_langchain",
    "custom_retriever_all_sources",
    "mongo_db",
    "CONCURRENCY_COUNT",
    "AVAILABLE_SOURCES_UI",
    "AVAILABLE_SOURCES",
]
