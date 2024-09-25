import pdb

import gradio as gr
import logfire
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.llms import MessageRole
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.tools import RetrieverTool, ToolMetadata
from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.llms.openai import OpenAI
from prompts import system_message_openai_agent
from setup import (  # custom_retriever_langchain,; custom_retriever_llama_index,; custom_retriever_openai_cookbooks,; custom_retriever_peft,; custom_retriever_transformers,; custom_retriever_trl,
    AVAILABLE_SOURCES,
    AVAILABLE_SOURCES_UI,
    CONCURRENCY_COUNT,
    custom_retriever_all_sources,
)


def update_query_engine_tools(selected_sources):
    tools = []
    source_mapping = {
        # "Transformers Docs": (
        #     custom_retriever_transformers,
        #     "Transformers_information",
        #     """Useful for general questions asking about the artificial intelligence (AI) field. Employ this tool to fetch information on topics such as language models (LLMs) models such as Llama3 and theory (transformer architectures), tips on prompting, quantization, etc.""",
        # ),
        # "PEFT Docs": (
        #     custom_retriever_peft,
        #     "PEFT_information",
        #     """Useful for questions asking about efficient LLM fine-tuning. Employ this tool to fetch information on topics such as LoRA, QLoRA, etc.""",
        # ),
        # "TRL Docs": (
        #     custom_retriever_trl,
        #     "TRL_information",
        #     """Useful for questions asking about fine-tuning LLMs with reinforcement learning (RLHF). Includes information about the Supervised Fine-tuning step (SFT), Reward Modeling step (RM), and the Proximal Policy Optimization (PPO) step.""",
        # ),
        # "LlamaIndex Docs": (
        #     custom_retriever_llama_index,
        #     "LlamaIndex_information",
        #     """Useful for questions asking about retrieval augmented generation (RAG) with LLMs and embedding models. It is the documentation of a framework, includes info about fine-tuning embedding models, building chatbots, and agents with llms, using vector databases, embeddings, information retrieval with cosine similarity or bm25, etc.""",
        # ),
        # "OpenAI Cookbooks": (
        #     custom_retriever_openai_cookbooks,
        #     "openai_cookbooks_info",
        #     """Useful for questions asking about accomplishing common tasks with the OpenAI API. Returns example code and guides stored in Jupyter notebooks, including info about ChatGPT GPT actions, OpenAI Assistants API,  and How to fine-tune OpenAI's GPT-4o and GPT-4o-mini models with the OpenAI API.""",
        # ),
        # "LangChain Docs": (
        #     custom_retriever_langchain,
        #     "langchain_info",
        #     """Useful for questions asking about the LangChain framework. It is the documentation of the LangChain framework, includes info about building chains, agents, and tools, using memory, prompts, callbacks, etc.""",
        # ),
        "All Sources": (
            custom_retriever_all_sources,
            "all_sources_info",
            """Useful for questions asking about information in the field of AI.""",
        ),
    }

    for source in selected_sources:
        if source in source_mapping:
            retriever, name, description = source_mapping[source]
            tools.append(
                RetrieverTool(
                    retriever=retriever,
                    metadata=ToolMetadata(
                        name=name,
                        description=description,
                    ),
                )
            )

    return tools


def generate_completion(
    query,
    history,
    sources,
    model,
    memory,
):
    with logfire.span("Running query"):
        logfire.info(f"User query: {query}")

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

        llm = OpenAI(temperature=1, model=model, max_tokens=None)
        client = llm._get_client()
        logfire.instrument_openai(client)

        query_engine_tools = update_query_engine_tools(["All Sources"])

        filter_list = []
        source_mapping = {
            "Transformers Docs": "transformers",
            "PEFT Docs": "peft",
            "TRL Docs": "trl",
            "LlamaIndex Docs": "llama_index",
            "LangChain Docs": "langchain",
            "OpenAI Cookbooks": "openai_cookbooks",
            "Towards AI Blog": "tai_blog",
        }

        for source in sources:
            if source in source_mapping:
                filter_list.append(
                    MetadataFilter(
                        key="source",
                        operator=FilterOperator.EQ,
                        value=source_mapping[source],
                    )
                )

        filters = MetadataFilters(
            filters=filter_list,
            condition=FilterCondition.OR,
        )
        query_engine_tools[0].retriever._vector_retriever._filters = filters

        agent = OpenAIAgent.from_tools(
            llm=llm,
            memory=memory,
            tools=query_engine_tools,
            system_prompt=system_message_openai_agent,
        )

    completion = agent.stream_chat(query)

    answer_str = ""
    for token in completion.response_gen:
        answer_str += token
        yield answer_str

    for answer_str in add_sources(answer_str, completion):
        yield answer_str


def add_sources(answer_str, completion):
    if completion is None:
        yield answer_str

    formatted_sources = format_sources(completion)
    if formatted_sources == "":
        yield answer_str

    if formatted_sources != "":
        answer_str += "\n\n" + formatted_sources

    yield answer_str


def format_sources(completion) -> str:
    if len(completion.sources) == 0:
        return ""

    logfire.info(f"Formatting sources: {completion.sources}")

    display_source_to_ui = {
        src: ui for src, ui in zip(AVAILABLE_SOURCES, AVAILABLE_SOURCES_UI)
    }

    documents_answer_template: str = (
        "📝 Here are the sources I used to answer your question:\n{documents}"
    )
    document_template: str = "[🔗 {source}: {title}]({url}), relevance: {score:2.2f}"
    all_documents = []
    for source in completion.sources:  # looping over list[ToolOutput]
        if isinstance(source.raw_output, Exception):
            logfire.error(f"Error in source output: {source.raw_output}")
            # pdb.set_trace()
            continue

        if not isinstance(source.raw_output, list):
            logfire.warn(f"Unexpected source output type: {type(source.raw_output)}")
            continue
        for src in source.raw_output:  # looping over list[NodeWithScore]
            document = document_template.format(
                title=src.metadata["title"],
                score=src.score,
                source=display_source_to_ui.get(
                    src.metadata["source"], src.metadata["source"]
                ),
                url=src.metadata["url"],
            )
            all_documents.append(document)

    if len(all_documents) == 0:
        return ""
    else:
        documents = "\n".join(all_documents)
        return documents_answer_template.format(documents=documents)


def save_completion(completion, history):
    pass


def vote(data: gr.LikeData):
    pass


accordion = gr.Accordion(label="Customize Sources (Click to expand)", open=False)
sources = gr.CheckboxGroup(
    AVAILABLE_SOURCES_UI,
    label="Sources",
    value=[
        "Transformers Docs",
        "PEFT Docs",
        "TRL Docs",
        "LlamaIndex Docs",
        "LangChain Docs",
        "OpenAI Cookbooks",
        # "All Sources",
    ],
    interactive=True,
)
model = gr.Dropdown(
    [
        "gpt-4o-mini",
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

    memory = gr.State(
        lambda: ChatSummaryMemoryBuffer.from_defaults(
            token_limit=120000,
        )
    )
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
