"""
Integration tests for FalkorDB vector store functionality.

These tests validate the end-to-end process of constructing, indexing,
and searching vector embeddings in a FalkorDB instance. They include:
- Setting up the FalkorDB vector store with a local instance.
- Indexing documents with fake embeddings.
- Performing vector searches and validating results.

Note:
These tests are conducted using a local FalkorDB instance but can also
be run against a Cloud FalkorDB instance. Ensure that appropriate host
and port configurations are set up before running the tests.
"""

import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_falkordb.vectorstores import (
    FalkorDBVector,
    process_index_data,
)

from typing import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import ReadWriteTestSuite


# Load environment variables from .env file
load_dotenv()

host = os.getenv("FALKORDB_HOST", "localhost")
port = int(os.getenv("FALKORDB_PORT", 6379))

OS_TOKEN_COUNT = 1535

texts = ["foo", "bar", "baz", "It is the end of the world. Take shelter!", "new foo"]





class TestChromaStandard(ReadWriteTestSuite):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        store = FalkorDBVector(
            embedding=FakeEmbeddingsWithOsDimension(),
            host=host,
            port=port,
        ) 
        store.add_documents(
            [
        Document(
            metadata={
                "text": "foo",
                "id": "acbd18db4cc2f85cedef654fccc4a4d8",
                "page": "0",
                "source": "source"
            },
            page_content="foo",
        )
    ]
        )
        store.delete()
        try:
            yield store
        finally:
            store.delete()
            pass
    
    @pytest.mark.xfail(reason=("get_by_ids not implemented."))
    def test_get_by_ids(self, vectorstore: VectorStore) -> None:
        super().test_get_by_ids(vectorstore)

    @pytest.mark.xfail(reason=("get_by_ids not implemented."))
    def test_add_documents_documents(self, vectorstore: VectorStore) -> None:
        super().test_add_documents_documents(vectorstore)

    @pytest.mark.xfail(reason=("get_by_ids not implemented."))
    def test_get_by_ids_missing(self, vectorstore: VectorStore) -> None:
        super().test_get_by_ids_missing(vectorstore)

    @pytest.mark.xfail(reason=("get_by_ids not implemented."))
    def test_add_documents_with_existing_ids(self, vectorstore: VectorStore) -> None:
        super().test_add_documents_with_existing_ids(vectorstore)



class FakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings.
        Embeddings encode each text as its index."""
        return [[float(1.0)] * 9 + [float(i)] for i in range(len(texts))]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Return constant query embeddings.
        Embeddings are identical to embed_documents(texts)[0].
        Distance to each text will be that text's index,
        as it was passed to embed_documents."""
        return [float(1.0)] * 9 + [float(0.0)]

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


def drop_vector_indexes(store: FalkorDBVector) -> None:
    """Cleanup all vector indexes"""
    index_entity_labels: List[Any] = []
    index_entity_properties: List[Any] = []
    index_entity_types: List[Any] = []

    # get all indexes
    result = store._query(
        """
    CALL db.indexes()
    """
    )
    processed_result: List[Dict[str, Any]] = process_index_data(result)

    # get all vector indexs entity labels, entity properties, entity_types
    if isinstance(processed_result, list):
        for index in processed_result:
            if isinstance(index, dict):
                if index.get("index_type") == "VECTOR":
                    index_entity_labels.append(index["entity_label"])
                    index_entity_properties.append(index["entity_property"])
                    index_entity_types.append(index["entity_type"])

    # drop vector indexs
    for entity_label, entity_property, entity_type in zip(
        index_entity_labels, index_entity_properties, index_entity_types
    ):
        if entity_type == "NODE":
            store._database.drop_node_vector_index(
                label=entity_label,
                attribute=entity_property,
            )
        elif entity_type == "RELATIONSHIP":
            store._database.drop_edge_vector_index(
                label=entity_label,
                attribute=entity_property,
            )


class FakeEmbeddingsWithOsDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, embedding_texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (OS_TOKEN_COUNT - 1) + [float(i + 1)]
            for i in range(len(embedding_texts))
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (OS_TOKEN_COUNT - 1) + [float(texts.index(text) + 1)]
