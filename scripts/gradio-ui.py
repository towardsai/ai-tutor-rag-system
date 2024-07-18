import json
import logging
import os
import pickle
from datetime import datetime
from typing import Optional

import chromadb
import gradio as gr
from custom_retriever import CustomRetriever
from dotenv import load_dotenv
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.agent import AgentRunner, ReActAgent
from llama_index.core.chat_engine import (
    CondensePlusContextChatEngine,
    CondenseQuestionChatEngine,
    ContextChatEngine,
)
from llama_index.core.data_structs import Node
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import BaseNode, MetadataMode, NodeWithScore, TextNode
from llama_index.core.tools import (
    FunctionTool,
    QueryEngineTool,
    RetrieverTool,
    ToolMetadata,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.utils import GPT4_MODELS
from llama_index.vector_stores.chroma import ChromaVectorStore
from tutor_prompts import (
    TEXT_QA_TEMPLATE,
    QueryValidation,
    system_message_openai_agent,
    system_message_validation,
    system_prompt,
)

load_dotenv(".env")


# from utils import init_mongo_db

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("gradio").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# # This variables are used to intercept API calls
# # launch mitmweb
# cert_file = "/Users/omar/Documents/mitmproxy-ca-cert.pem"
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


from llama_index.llms.openai.utils import (
    ALL_AVAILABLE_MODELS,
    AZURE_TURBO_MODELS,
    CHAT_MODELS,
    GPT3_5_MODELS,
    GPT3_MODELS,
    GPT4_MODELS,
    TURBO_MODELS,
)

# Add new models to GPT4_MODELS
new_gpt4_models = {
    "gpt-4-1106-preview": 128000,
    "gpt-4-0125-preview": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4-turbo-2024-04-09": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4o-2024-05-13": 128000,
    "gpt-4o-mini": 128000,
}
GPT4_MODELS.update(new_gpt4_models)

# Update ALL_AVAILABLE_MODELS
ALL_AVAILABLE_MODELS.update(new_gpt4_models)

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
vector_retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
    use_async=True,
    embed_model=OpenAIEmbedding(model="text-embedding-3-large", mode="similarity"),
)

memory = ChatMemoryBuffer.from_defaults(token_limit=150000)


with open("scripts/ai-tutor-vector-db/document_dict.pkl", "rb") as f:
    document_dict = pickle.load(f)

custom_retriever = CustomRetriever(vector_retriever, document_dict)


def format_sources(completion) -> str:
    if len(completion.sources) == 0:
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
            for src in completion.sources[0].raw_output
        ]
    )

    return documents_answer_template.format(documents=documents)


def add_sources(answer_str, completion):
    if completion is None:
        yield answer_str

    formatted_sources = format_sources(completion)
    if formatted_sources == "":
        yield answer_str

    if formatted_sources != "":
        answer_str += "\n\n" + formatted_sources

    yield answer_str


def generate_completion(
    query,
    history,
    sources,
    model,
):

    print(f"query: {query}")
    print(model)
    print(sources)

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

    # completion = response_synthesizer.synthesize(query, nodes=nodes_context)
    custom_query_engine = RetrieverQueryEngine(
        retriever=custom_retriever,
        response_synthesizer=response_synthesizer,
    )

    # agent = CondensePlusContextChatEngine.from_defaults(
    # agent = CondenseQuestionChatEngine.from_defaults(

    # agent = ContextChatEngine.from_defaults(
    #     retriever=custom_retriever,
    #     context_template=system_prompt,
    #     llm=llm,
    #     memory=memory,
    #     verbose=True,
    # )

    query_engine_tools = [
        RetrieverTool(
            retriever=custom_retriever,
            metadata=ToolMetadata(
                name="AI_information",
                description="""Only use this tool if necessary. The 'AI_information' tool is a comprehensive repository for information in artificial intelligence (AI). When using this tool, the input should be the user's question rewritten as a statement. e.g. When the user asks 'How can I fine-tune an LLM?', the input should be 'Fine-tune an Large Language Model (LLM)'. The input can also be adapted to focus on specific aspects or further details of the current topic under discussion. This dynamic input approach allows for a tailored exploration of AI subjects, ensuring that responses are relevant and informative. Employ this tool to fetch nuanced information on topics such as model training, fine-tuning, and LLM augmentation, thereby facilitating a rich, context-aware dialogue. """,
            ),
        )
    ]

    if model == "gemini-1.5-flash" or model == "gemini-1.5-pro":
        # agent = AgentRunner.from_llm(
        #     llm=llm,
        #     tools=query_engine_tools,
        #     verbose=True,
        #     memory=memory,
        #     # system_prompt=system_message_openai_agent,
        # )
        agent = ReActAgent.from_tools(
            llm=llm,
            memory=memory,
            tools=query_engine_tools,
            verbose=True,
            # system_prompt=system_message_openai_agent,
        )
        prompts = agent._get_prompt_modules()
        print(prompts.values())
    else:
        agent = OpenAIAgent.from_tools(
            llm=llm,
            memory=memory,
            tools=query_engine_tools,
            verbose=True,
            system_prompt=system_message_openai_agent,
        )

    # completion = custom_query_engine.query(query)
    completion = agent.stream_chat(query)

    # completion = agent.chat(query)
    # return str(completion)

    answer_str = ""
    for token in completion.response_gen:
        answer_str += token
        yield answer_str

    logger.info(f"completion: {answer_str=}")

    for sources in add_sources(answer_str, completion):
        yield sources

    logger.info(f"source: {sources=}")


def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value["value"])
    else:
        print("You downvoted this response: " + data.value["value"])


accordion = gr.Accordion(label="Customize Sources (Click to expand)", open=False)
sources = gr.CheckboxGroup(
    AVAILABLE_SOURCES_UI, label="Sources", value="HF Transformers", interactive=False
)
model = gr.Dropdown(
    [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gpt-3.5-turbo",
        "gpt-4o-mini",
        "gpt-4o",
    ],
    label="Model",
    value="gpt-4o-mini",
    interactive=True,
)

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
