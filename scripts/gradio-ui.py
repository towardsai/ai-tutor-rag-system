import os
import pickle

import chromadb
import gradio as gr
import logfire
from custom_retriever import CustomRetriever
from dotenv import load_dotenv
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.tools import RetrieverTool, ToolMetadata
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from tutor_prompts import system_message_openai_agent

# from utils import init_mongo_db

load_dotenv()

logfire.configure()


CONCURRENCY_COUNT = int(os.getenv("CONCURRENCY_COUNT", 64))
MONGODB_URI = os.getenv("MONGODB_URI")

DB_PATH = os.getenv("DB_PATH", f"scripts/ai-tutor-vector-db")
DB_COLLECTION = os.getenv("DB_NAME", "ai-tutor-vector-db")

if not os.path.exists(DB_PATH):
    # Download the vector database from the Hugging Face Hub if it doesn't exist locally
    # https://huggingface.co/datasets/towardsai-buster/ai-tutor-db/tree/main
    logfire.warn(
        f"Vector database does not exist at {DB_PATH}, downloading from Hugging Face Hub"
    )
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="towardsai-buster/ai-tutor-vector-db",
        local_dir=DB_PATH,
        repo_type="dataset",
    )
    logfire.info(f"Downloaded vector database to {DB_PATH}")

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
vector_retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
    use_async=True,
    embed_model=OpenAIEmbedding(model="text-embedding-3-large", mode="similarity"),
)
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
    memory,
):

    with logfire.span("Running query"):
        logfire.info(f"query: {query}")
        logfire.info(f"model: {model}")
        logfire.info(f"sources: {sources}")

        chat_list = memory.get()

        if len(chat_list) != 0:
            user_index = [
                i for i, msg in enumerate(chat_list) if msg.role == MessageRole.USER
            ]
            if len(user_index) > len(history):
                user_index_to_remove = user_index[len(history)]
                chat_list = chat_list[:user_index_to_remove]
                memory.set(chat_list)

        logfire.info(f"chat_history: {len(memory.get())} {memory.get()}")
        logfire.info(f"gradio_history: {len(history)} {history}")

        # # # TODO: change source UI name to actual source name
        # filters = MetadataFilters(
        #     filters=[
        #         # MetadataFilter(key="source", value="HF_Transformers"),
        #         # MetadataFilter(key="source", value="towards_ai_blog"),
        #         MetadataFilter(
        #             key="source", operator=FilterOperator.EQ, value="HF_Transformers"
        #         ),
        #     ],
        #     # condition=FilterCondition.OR,
        # )
        # vector_retriever = VectorIndexRetriever(
        #     # filters=filters,
        #     index=index,
        #     similarity_top_k=10,
        #     use_async=True,
        #     embed_model=OpenAIEmbedding(model="text-embedding-3-large", mode="similarity"),
        # )
        # custom_retriever = CustomRetriever(vector_retriever, document_dict)

        llm = OpenAI(temperature=1, model=model, max_tokens=None)
        client = llm._get_client()
        logfire.instrument_openai(client)

        query_engine_tools = [
            RetrieverTool(
                retriever=custom_retriever,
                metadata=ToolMetadata(
                    name="AI_information",
                    description="""Only use this tool if necessary. The 'AI_information' tool returns information about the artificial intelligence (AI) field. When using this tool, the input should be the user's question rewritten as a statement. e.g. When the user asks 'How can I quantize a model?', the input should be 'Model quantization'. The input can also be adapted to focus on specific aspects or further details of the current topic under discussion. This dynamic input approach allows for a tailored exploration of AI subjects, ensuring that responses are relevant and informative. Employ this tool to fetch nuanced information on topics such as model training, fine-tuning, and LLM augmentation, thereby facilitating a rich, context-aware dialogue. """,
                ),
            )
        ]

        agent = OpenAIAgent.from_tools(
            llm=llm,
            memory=memory,
            tools=query_engine_tools,  # type: ignore
            system_prompt=system_message_openai_agent,
        )

    completion = agent.stream_chat(query)

    answer_str = ""
    for token in completion.response_gen:
        answer_str += token
        yield answer_str

    for answer_str in add_sources(answer_str, completion):
        yield answer_str


def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value["value"])
    else:
        print("You downvoted this response: " + data.value["value"])


accordion = gr.Accordion(label="Customize Sources (Click to expand)", open=False)
sources = gr.CheckboxGroup(
    AVAILABLE_SOURCES_UI, label="Sources", value="HF Transformers", interactive=False  # type: ignore
)
model = gr.Dropdown(
    [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gpt-4o-mini",
        "gpt-4o",
    ],
    label="Model",
    value="gpt-4o-mini",
    interactive=False,
)

with gr.Blocks(
    fill_height=True,
    title="Towards AI 🤖",
    analytics_enabled=True,
) as demo:
    memory = gr.State(ChatMemoryBuffer.from_defaults(token_limit=120000))
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
        additional_inputs=[sources, model, memory],
        additional_inputs_accordion=accordion,
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=CONCURRENCY_COUNT)
    demo.launch(debug=False, share=False)
