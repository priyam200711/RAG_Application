import io
import json
import os
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from app.core.logging import logger


class DocumentLoaderService:
    """Loads various document formats (.txt, .md, .json, .pdf) into LangChain Documents."""

    @staticmethod
    def load_from_text(
        text: str,
        title: str,
        metadata: Optional[dict] = None
    ) -> Document:
        meta = metadata.copy() if metadata else {}
        meta["source"] = title
        meta["file_type"] = "raw_text"
        return Document(page_content=text, metadata=meta)

    @classmethod
    def load_from_bytes(
        cls,
        file_bytes: bytes,
        filename: str,
        metadata: Optional[dict] = None
    ) -> List[Document]:
        """Parse raw file bytes (uploaded via API or Web UI) into LangChain Documents."""
        base_meta = metadata.copy() if metadata else {}
        base_meta["source"] = filename
        suffix = Path(filename).suffix.lower()

        if suffix == ".pdf":
            try:
                import pypdf
            except ImportError:
                raise ImportError("pypdf is required to process PDF files. Run `pip install pypdf`.")

            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            docs = []
            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    page_meta = {
                        **base_meta,
                        "file_type": ".pdf",
                        "page_number": page_idx + 1,
                        "total_pages": len(reader.pages)
                    }
                    docs.append(Document(page_content=page_text, metadata=page_meta))

            logger.info("Parsed PDF '%s': %d pages extracted.", filename, len(docs))
            return docs

        elif suffix in [".txt", ".md", ".markdown"]:
            text = file_bytes.decode("utf-8", errors="replace")
            meta = {**base_meta, "file_type": suffix}
            return [Document(page_content=text, metadata=meta)]

        elif suffix == ".json":
            data = json.loads(file_bytes.decode("utf-8", errors="replace"))
            if isinstance(data, list):
                docs = []
                for idx, item in enumerate(data):
                    t = item.get("text") or item.get("content") or json.dumps(item, indent=2)
                    m = {**base_meta, "file_type": ".json", "item_index": idx}
                    docs.append(Document(page_content=str(t), metadata=m))
                return docs
            else:
                t = json.dumps(data, indent=2)
                return [Document(page_content=t, metadata={**base_meta, "file_type": ".json"})]

        else:
            text = file_bytes.decode("utf-8", errors="replace")
            return [Document(page_content=text, metadata={**base_meta, "file_type": suffix})]

    @classmethod
    def load_from_file(cls, file_path: str, title: Optional[str] = None) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found at: {file_path}")

        doc_title = title or path.name
        suffix = path.suffix.lower()

        if suffix in [".txt", ".md", ".markdown"]:
            return cls._load_text_file(path, doc_title)
        elif suffix == ".json":
            return cls._load_json_file(path, doc_title)
        elif suffix == ".pdf":
            return cls._load_pdf_file(path, doc_title)
        else:
            logger.warning("Unrecognized extension '%s', attempting text read.", suffix)
            return cls._load_text_file(path, doc_title)

    @classmethod
    def load_directory(
        cls,
        directory_path: str,
        recursive: bool = True
    ) -> List[Document]:
        dir_path = Path(directory_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")

        all_docs: List[Document] = []
        pattern = "**/*" if recursive else "*"
        supported_exts = {".txt", ".md", ".markdown", ".json", ".pdf"}

        for p in dir_path.glob(pattern):
            if p.is_file() and p.suffix.lower() in supported_exts:
                try:
                    docs = cls.load_from_file(str(p))
                    all_docs.extend(docs)
                except Exception as e:
                    logger.error("Failed to load file %s: %s", p, e)

        logger.info("Loaded %d documents from directory: %s", len(all_docs), directory_path)
        return all_docs

    @staticmethod
    def _load_text_file(path: Path, title: str) -> List[Document]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        metadata = {
            "source": title,
            "file_path": str(path.resolve()),
            "file_type": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
        }
        return [Document(page_content=content, metadata=metadata)]

    @staticmethod
    def _load_json_file(path: Path, title: str) -> List[Document]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)

        if isinstance(data, list):
            docs = []
            for idx, item in enumerate(data):
                text = item.get("text") or item.get("content") or json.dumps(item, indent=2)
                item_meta = {
                    "source": f"{title}#item_{idx}",
                    "file_path": str(path.resolve()),
                    "file_type": ".json",
                    "item_index": idx
                }
                docs.append(Document(page_content=str(text), metadata=item_meta))
            return docs
        else:
            text = json.dumps(data, indent=2)
            metadata = {
                "source": title,
                "file_path": str(path.resolve()),
                "file_type": ".json",
            }
            return [Document(page_content=text, metadata=metadata)]

    @staticmethod
    def _load_pdf_file(path: Path, title: str) -> List[Document]:
        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf is required to process PDF files. "
                "Install it with `pip install pypdf`."
            )

        reader = pypdf.PdfReader(str(path))
        docs = []
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                metadata = {
                    "source": f"{title} (page {page_idx + 1})",
                    "file_path": str(path.resolve()),
                    "file_type": ".pdf",
                    "page_number": page_idx + 1,
                    "total_pages": len(reader.pages)
                }
                docs.append(Document(page_content=page_text, metadata=metadata))
        return docs
