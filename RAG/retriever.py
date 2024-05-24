from typing import List, Optional

from datasets import Dataset
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import BaseRetriever
from llama_index.legacy.embeddings import HuggingFaceEmbedding
from llama_index.core import Settings

from dataloaders.common_utils import load_and_process_twitter_data
from document_scraper import scrape_documents

class VectorDBRetriever(BaseRetriever):
    def __init__(
        self,
        documents,
        embed_model,
        query_mode: str = "default",
        similarity_top_k: int = 3,
        generate_vector_store: bool = True,
    ):
        self._embed_model = embed_model
        Settings.embed_model = embed_model
        self.service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)

        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k

        if generate_vector_store:
            self.index = self.build_vector_store(documents)
        else:
            storage_context = StorageContext.from_defaults(persist_dir="../documents")
            self.index = load_index_from_storage(storage_context)

        self.query_engine = self.index.as_retriever(
            similarity_top_k=self._similarity_top_k,
        )
        super().__init__()

    def build_vector_store(self, documents):
        """
        Given a list of documents and a text encoder, we build a llama-index vector store index
        """
        text_chunks = []
        doc_idxs = []
        text_parser = SentenceSplitter(chunk_size=512)
        for doc_idx, doc in enumerate(documents):
            cur_text_chunks = text_parser.split_text(doc.text)
            text_chunks.extend(cur_text_chunks)
            doc_idxs.extend([doc_idx] * len(cur_text_chunks))

        nodes = []
        for idx, text_chunk in enumerate(text_chunks):
            node = TextNode(text=text_chunk)
            src_doc = documents[doc_idxs[idx]]
            node.metadata = src_doc.metadata
            nodes.append(node)
        for node in nodes:
            node_embedding = self._embed_model.get_text_embedding(
                node.get_content(metadata_mode="all")
            )
            node.embedding = node_embedding

        index = VectorStoreIndex(nodes, service_context=self.service_context)
        index.storage_context.persist(persist_dir="../documents")
        return index

    def _retrieve(self, query_str) -> List[NodeWithScore]:
        query_result = self.query_engine.retrieve(query_str)
        return query_result

    def get_added_message(self, query_str) -> str:
        query_result = self._retrieve(query_str)

        start = "You are given the following context:"
        context = ""
        for cur_node in query_result:
            context += cur_node.get_content()
        context += '\n'
        question = "Is the following hate speech: "

        return start + context + question


def augment_dataset(dataset):
    documents = scrape_documents()
    embed_model = HuggingFaceEmbedding(
        model_name="distilbert/distilbert-base-uncased"
    )
    retriever = VectorDBRetriever(documents, embed_model, generate_vector_store=True)

    dataset = dataset.to_pandas()
    dataset['text'] = dataset['text'].apply(lambda q: retriever.get_added_message(q))
    dataset = Dataset.from_pandas(dataset)

    return dataset

