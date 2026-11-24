import unittest
from app.api.routes.documents import clear_documents, get_document_stats, ingest_text
from app.api.routes.health import get_health
from app.api.routes.rag import query_rag
from app.api.schemas import IngestTextRequest, QueryRequest
from app.services.vector_store import vector_store_service


class TestRAGApiEndpoints(unittest.TestCase):
    def setUp(self):
        vector_store_service.clear()

    def tearDown(self):
        vector_store_service.clear()

    def test_health_endpoint(self):
        health = get_health()
        self.assertEqual(health.status, "healthy")
        self.assertTrue(len(health.app_name) > 0)
        self.assertTrue(len(health.llm_provider) > 0)
        self.assertEqual(health.total_indexed_chunks, 0)

    def test_document_ingestion_and_stats(self):
        payload = IngestTextRequest(
            title="OAuth Security Guide",
            content="OAuth 2.0 uses bearer tokens and refresh tokens to grant authorization without exposing credentials.",
            metadata={"author": "Security Team"}
        )
        res_ingest = ingest_text(payload)
        self.assertEqual(res_ingest.status, "success")
        self.assertGreater(res_ingest.total_chunks_created, 0)

        stats = get_document_stats()
        self.assertGreaterEqual(stats.total_chunks, 1)
        self.assertIn("OAuth Security Guide", stats.unique_sources)

    def test_rag_query_grounded_answer(self):
        payload = IngestTextRequest(
            title="Database Policy",
            content="All PostgreSQL production databases must use read replicas for reporting queries.",
            metadata={}
        )
        ingest_text(payload)

        query_payload = QueryRequest(
            query="What are read replicas used for in databases?",
            top_k=2
        )
        res_query = query_rag(query_payload)
        self.assertIn("read replicas", res_query.answer.lower())
        self.assertGreater(len(res_query.citations), 0)
        self.assertEqual(res_query.citations[0].source, "Database Policy")

    def test_anti_hallucination_on_unindexed_topic(self):
        query_payload = QueryRequest(query="What is the distance to Alpha Centauri?")
        res_query = query_rag(query_payload)
        self.assertIn("do not have enough information", res_query.answer.lower())

    def test_clear_documents(self):
        payload = IngestTextRequest(title="Temp Doc", content="Temporary testing document content.")
        ingest_text(payload)
        self.assertGreater(get_document_stats().total_chunks, 0)

        res_clear = clear_documents()
        self.assertEqual(res_clear["status"], "success")
        self.assertEqual(get_document_stats().total_chunks, 0)


if __name__ == "__main__":
    unittest.main()
