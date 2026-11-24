import unittest
from app.services.document_loader import DocumentLoaderService
from app.services.text_splitter import DocumentChunker


class TestDocumentIngestion(unittest.TestCase):
    def setUp(self):
        self.chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

    def test_text_splitting_and_metadata(self):
        sample_text = (
            "Retrieval-Augmented Generation (RAG) optimizes the output of an LLM. "
            "It references an authoritative knowledge base outside of its training data sources. "
            "FastAPI provides high performance and modern web standards."
        )
        chunks = self.chunker.split_text(sample_text, source="test_doc.txt", metadata={"department": "AI"})
        
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].metadata["source"], "test_doc.txt")
        self.assertEqual(chunks[0].metadata["department"], "AI")
        self.assertEqual(chunks[0].metadata["chunk_id"], 0)
        self.assertEqual(chunks[0].metadata["total_chunks"], len(chunks))

    def test_load_from_text(self):
        doc = DocumentLoaderService.load_from_text(
            text="LangChain framework components.",
            title="LangChain Intro",
            metadata={"version": "1.0"}
        )
        self.assertEqual(doc.page_content, "LangChain framework components.")
        self.assertEqual(doc.metadata["source"], "LangChain Intro")
        self.assertEqual(doc.metadata["version"], "1.0")


if __name__ == "__main__":
    unittest.main()
