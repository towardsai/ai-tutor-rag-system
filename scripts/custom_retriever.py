import logging
import time
from typing import List

import logfire
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.postprocessor.cohere_rerank import CohereRerank

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        document_dict: dict,
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._document_dict = document_dict
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        # LlamaIndex adds "\ninput is " to the query string
        query_bundle.query_str = query_bundle.query_str.replace("\ninput is ", "")
        query_bundle.query_str = query_bundle.query_str.rstrip()

        logfire.info(f"Retrieving 10 nodes with string: '{query_bundle}'")
        start = time.time()
        nodes = self._vector_retriever.retrieve(query_bundle)

        duration = time.time() - start
        logfire.info(f"Retrieving nodes took {duration:.2f}s")

        # Filter out nodes with the same ref_doc_id
        def filter_nodes_by_unique_doc_id(nodes):
            unique_nodes = {}
            for node in nodes:
                doc_id = node.node.ref_doc_id
                if doc_id is not None and doc_id not in unique_nodes:
                    unique_nodes[doc_id] = node
            return list(unique_nodes.values())

        nodes = filter_nodes_by_unique_doc_id(nodes)
        logfire.info(f"Number of nodes after filtering: {len(nodes)}")

        nodes_context = []
        for node in nodes:
            # print("Node ID\t", node.node_id)
            # print("Title\t", node.metadata["title"])
            # print("Text\t", node.text)
            # print("Score\t", node.score)
            # print("Metadata\t", node.metadata)
            # print("-_" * 20)
            if node.metadata["retrieve_doc"] == True:
                # print("This node will be replaced by the document")
                doc = self._document_dict[node.node.ref_doc_id]
                # print(doc.text)
                new_node = NodeWithScore(
                    node=TextNode(text=doc.text, metadata=node.metadata),  # type: ignore
                    score=node.score,
                )
                nodes_context.append(new_node)
            else:
                nodes_context.append(node)

        reranker = CohereRerank(top_n=8, model="rerank-english-v3.0")
        nodes_context = reranker.postprocess_nodes(nodes_context, query_bundle)
        logfire.info(f"Cohere raranking to {len(nodes_context)} nodes")

        return nodes_context
