from typing import List, Optional
from llama_index.core import SummaryIndex, Document, QueryBundle, VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import BaseRetriever


def scrape_documents():
    hate_speech_explanations = [
        "https://www.coe.int/en/web/combating-hate-speech/what-is-hate-speech-and-why-is-it-a-problem-",
        "https://en.wikipedia.org/wiki/Hate_speech",
        "https://plato.stanford.edu/entries/hate-speech/",
        "https://www.un.org/en/hate-speech/understanding-hate-speech/what-is-hate-speech",
        "https://www.ala.org/advocacy/intfreedom/hate",
        "https://www.rightsforpeace.org/hate-speech",
        "https://transparency.meta.com/policies/community-standards/hate-speech/",
        "https://challengehate.com/what-is-hate-speech/",
        "https://inhope.org/EN/articles/what-is-hate-speech",
        "https://www.article19.org/resources/hate-speech-explained-a-summary/",
        "https://about.fb.com/news/2017/06/hard-questions-hate-speech/",
        "https://www.cilvektiesibugids.lv/en/themes/freedom-of-expression-media/freedom-of-expression/hate-speech/what-is-hate-speech",
        "https://items.ssrc.org/disinformation-democracy-and-conflict-prevention/classifying-and-identifying-the-intensity-of-hate-speech/"
    ]

    documents = SimpleWebPageReader(html_to_text=True).load_data(
        hate_speech_explanations
    )

    print(documents[0])
    return documents


class VectorDBRetriever(BaseRetriever):
    def __init__(
        self,
        documents,
        embed_model,
        query_mode: str = "default",
        similarity_top_k: int = 3,
    ):
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        self.index = self.build_vector_store(documents)
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=self._similarity_top_k,
        )
        super().__init__()

    def build_vector_store(self, documents):
        """
        Given a list of documents and a text encoder, we build llama-index vector store
        """
        text_chunks = []
        doc_idxs = []
        text_parser = SentenceSplitter(chunk_size=1024)
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

        index = VectorStoreIndex(nodes)
        return index

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(
            query_bundle.query_str
        )
        query_result = self.query_engine.query(query_embedding)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores


scrape_documents()