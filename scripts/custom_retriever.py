import asyncio
import time
import traceback
from typing import List, Optional

import logfire
import tiktoken
from cohere import AsyncClient
from llama_index.core import QueryBundle
from llama_index.core.async_utils import run_async_tasks
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.cohere_rerank.base import CohereRerank


class AsyncCohereRerank(CohereRerank):
    def __init__(
        self,
        top_n: int = 5,
        model: str = "rerank-english-v3.0",
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(top_n=top_n, model=model, api_key=api_key)
        self._api_key = api_key
        self._model = model
        self._top_n = top_n

    async def apostprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")

        if len(nodes) == 0:
            return []

        async_client = AsyncClient(api_key=self._api_key)

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self._model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self._top_n,
            },
        ) as event:
            texts = [
                node.node.get_content(metadata_mode=MetadataMode.EMBED)
                for node in nodes
            ]

            results = await async_client.rerank(
                model=self._model,
                top_n=self._top_n,
                query=query_bundle.query_str,
                documents=texts,
            )

            new_nodes = []
            for result in results.results:
                new_node_with_score = NodeWithScore(
                    node=nodes[result.index].node, score=result.relevance_score
                )
                new_nodes.append(new_node_with_score)
            event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes


class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        document_dict: dict,
        keyword_retriever,
        mode: str = "AND",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._document_dict = document_dict

        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode

        super().__init__()

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        # LlamaIndex adds "\ninput is " to the query string
        query_bundle.query_str = query_bundle.query_str.replace("\ninput is ", "")
        query_bundle.query_str = query_bundle.query_str.rstrip()

        # logfire.info(f"Retrieving nodes with string: '{query_bundle}'")
        start = time.time()
        nodes = await self._vector_retriever.aretrieve(query_bundle)
        keyword_nodes = await self._keyword_retriever.aretrieve(query_bundle)

        # logfire.info(f"Number of vector nodes: {len(nodes)}")
        # logfire.info(f"Number of keyword nodes: {len(keyword_nodes)}")

        vector_ids = {n.node.node_id for n in nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        nodes = [combined_dict[rid] for rid in retrieve_ids]

        # Filter out nodes with the same ref_doc_id
        def filter_nodes_by_unique_doc_id(nodes):
            unique_nodes = {}
            for node in nodes:
                # doc_id = node.node.ref_doc_id
                doc_id = node.node.source_node.node_id
                if doc_id is not None and doc_id not in unique_nodes:
                    unique_nodes[doc_id] = node
            return list(unique_nodes.values())

        nodes = filter_nodes_by_unique_doc_id(nodes)
        # logfire.info(
        #     f"Number of nodes after filtering the ones with same ref_doc_id: {len(nodes)}"
        # )
        # logfire.info(f"Nodes retrieved: {nodes}")

        nodes_context = []
        for node in nodes:
            # print("Node ID\t", node.node_id)
            # print("Title\t", node.metadata["title"])
            # print("Text\t", node.text)
            # print("Score\t", node.score)
            # print("Metadata\t", node.metadata)
            # print("-_" * 20)
            doc_id = node.node.source_node.node_id  # type: ignore
            if node.metadata["retrieve_doc"] == True:
                # print("This node will be replaced by the document")
                # doc = self._document_dict[node.node.ref_doc_id]
                # print("retrieved doc == True")
                doc = self._document_dict[doc_id]
                # print(doc.text)
                new_node = NodeWithScore(
                    node=TextNode(text=doc.text, metadata=node.metadata, id_=doc_id),  # type: ignore
                    score=node.score,
                )
                nodes_context.append(new_node)
            else:
                node.node.node_id = doc_id
                nodes_context.append(node)

        try:
            reranker = AsyncCohereRerank(top_n=3, model="rerank-english-v3.0")
            nodes_context = await reranker.apostprocess_nodes(
                nodes_context, query_bundle
            )

        except Exception as e:
            error_msg = f"Error during reranking: {type(e).__name__}: {str(e)}\n"
            error_msg += "Traceback:\n"
            error_msg += traceback.format_exc()
            logfire.error(error_msg)

        nodes_filtered = []
        total_tokens = 0
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        for node in nodes_context:
            if node.score < 0.10:  # type: ignore
                continue

            # Count tokens
            if "tokens" in node.node.metadata:
                node_tokens = node.node.metadata["tokens"]
            else:
                node_tokens = len(enc.encode(node.node.text))  # type: ignore

            if total_tokens + node_tokens > 100_000:
                logfire.info("Skipping node due to token count exceeding 100k")
                break

            total_tokens += node_tokens
            nodes_filtered.append(node)

        # logfire.info(f"Final nodes to context {len(nodes_filtered)} nodes")
        # logfire.info(f"Total tokens: {total_tokens}")

        # duration = time.time() - start
        # logfire.info(f"Retrieving nodes took {duration:.2f}s")

        return nodes_filtered[:3]

    # def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
    #     return asyncio.run(self._aretrieve(query_bundle))

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        # LlamaIndex adds "\ninput is " to the query string
        query_bundle.query_str = query_bundle.query_str.replace("\ninput is ", "")
        query_bundle.query_str = query_bundle.query_str.rstrip()
        logfire.info(f"Retrieving nodes with string: '{query_bundle}'")

        start = time.time()
        nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        logfire.info(f"Number of vector nodes: {len(nodes)}")
        logfire.info(f"Number of keyword nodes: {len(keyword_nodes)}")

        vector_ids = {n.node.node_id for n in nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        nodes = [combined_dict[rid] for rid in retrieve_ids]

        def filter_nodes_by_unique_doc_id(nodes):
            unique_nodes = {}
            for node in nodes:
                # doc_id = node.node.ref_doc_id
                doc_id = node.node.source_node.node_id
                if doc_id is not None and doc_id not in unique_nodes:
                    unique_nodes[doc_id] = node
            return list(unique_nodes.values())

        nodes = filter_nodes_by_unique_doc_id(nodes)
        logfire.info(
            f"Number of nodes after filtering the ones with same ref_doc_id: {len(nodes)}"
        )
        logfire.info(f"Nodes retrieved: {nodes}")

        nodes_context = []
        for node in nodes:
            doc_id = node.node.source_node.node_id  # type: ignore
            if node.metadata["retrieve_doc"] == True:
                doc = self._document_dict[doc_id]
                new_node = NodeWithScore(
                    node=TextNode(text=doc.text, metadata=node.metadata, id_=doc_id),  # type: ignore
                    score=node.score,
                )
                nodes_context.append(new_node)
            else:
                node.node.node_id = doc_id
                nodes_context.append(node)

        try:
            reranker = CohereRerank(top_n=3, model="rerank-english-v3.0")
            nodes_context = reranker.postprocess_nodes(nodes_context, query_bundle)

        except Exception as e:
            error_msg = f"Error during reranking: {type(e).__name__}: {str(e)}\n"
            error_msg += "Traceback:\n"
            error_msg += traceback.format_exc()
            logfire.error(error_msg)

        nodes_filtered = []
        total_tokens = 0
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        for node in nodes_context:
            if node.score < 0.10:  # type: ignore
                continue
            if "tokens" in node.node.metadata:
                node_tokens = node.node.metadata["tokens"]
            else:
                node_tokens = len(enc.encode(node.node.text))  # type: ignore

            if total_tokens + node_tokens > 100_000:
                logfire.info("Skipping node due to token count exceeding 100k")
                break

            total_tokens += node_tokens
            nodes_filtered.append(node)

        logfire.info(f"Final nodes to context {len(nodes_filtered)} nodes")
        logfire.info(f"Total tokens: {total_tokens}")

        duration = time.time() - start
        logfire.info(f"Retrieving nodes took {duration:.2f}s")

        return nodes_filtered[:3]
