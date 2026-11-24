import unittest
from langchain_core.documents import Document
from app.services.vector_store import VectorStoreService


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.vs_service = VectorStoreService()
        self.vs_service.clear()

    def tearDown(self):
        self.vs_service.clear()

    def test_add_and_search_documents(self):
        docs = [
            Document(page_content="Python FastAPI is high performance.", metadata={"source": "fastapi_doc"}),
            Document(page_content="LangChain is a composable orchestration framework.", metadata={"source": "langchain_doc"})
        ]
        added = self.vs_service.add_documents(docs)
        self.assertEqual(added, 2)

        stats = self.vs_service.get_stats()
        self.assertEqual(stats["total_chunks"], 2)
        self.assertIn("fastapi_doc", stats["unique_sources"])
        self.assertIn("langchain_doc", stats["unique_sources"])

        # Search
        results = self.vs_service.similarity_search_with_score("FastAPI performance", k=1, threshold=0.0)
        self.assertGreater(len(results), 0)
        top_doc, score = results[0]
        self.assertEqual(top_doc.metadata["source"], "fastapi_doc")
        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
