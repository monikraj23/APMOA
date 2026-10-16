"""
APMOA Knowledge Manager

Manages persistent RAG knowledge documents.

Responsibilities
----------------
1. Store uploaded knowledge documents
2. Maintain document metadata
3. Register existing knowledge documents
4. List available documents
5. Retrieve stored document paths
6. Delete documents safely

Supported formats
-----------------
- TXT
- MD
- PDF
- DOCX

The actual embedding/vector-index logic remains inside the
existing APMOA RAG implementation.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "knowledge"
)

UPLOAD_DIR = (
    KNOWLEDGE_ROOT
    / "uploaded"
)

METADATA_DIR = (
    KNOWLEDGE_ROOT
    / "metadata"
)

METADATA_FILE = (
    METADATA_DIR
    / "documents.json"
)


# =====================================================================
# SUPPORTED FORMATS
# =====================================================================

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".docx",
}


# =====================================================================
# KNOWLEDGE MANAGER
# =====================================================================

class KnowledgeManager:
    """
    Persistent manager for APMOA RAG knowledge documents.
    """

    def __init__(
        self,
        upload_dir: Optional[Path] = None,
        metadata_file: Optional[Path] = None,
    ):

        self.upload_dir = Path(
            upload_dir
            if upload_dir is not None
            else UPLOAD_DIR
        )

        self.metadata_file = Path(
            metadata_file
            if metadata_file is not None
            else METADATA_FILE
        )

        self._initialize_storage()

    # =================================================================
    # INITIALIZATION
    # =================================================================

    def _initialize_storage(self):

        self.upload_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metadata_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.metadata_file.exists():

            self._write_metadata(
                {
                    "documents": []
                }
            )

    # =================================================================
    # METADATA
    # =================================================================

    def _read_metadata(
        self,
    ) -> Dict[str, Any]:

        if not self.metadata_file.exists():

            return {
                "documents": []
            }

        try:

            with open(
                self.metadata_file,
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

        except (
            json.JSONDecodeError,
            OSError,
        ):

            return {
                "documents": []
            }

        if not isinstance(
            data,
            dict,
        ):

            return {
                "documents": []
            }

        if not isinstance(
            data.get("documents"),
            list,
        ):

            data["documents"] = []

        return data

    def _write_metadata(
        self,
        data: Dict[str, Any],
    ):

        self.metadata_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_file = (
            self.metadata_file.with_suffix(
                ".tmp"
            )
        )

        with open(
            temporary_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )

        temporary_file.replace(
            self.metadata_file
        )

    # =================================================================
    # ID
    # =================================================================

    @staticmethod
    def _slugify(
        value: str,
    ) -> str:

        value = str(
            value
        ).strip().lower()

        value = re.sub(
            r"[^a-z0-9]+",
            "_",
            value,
        )

        value = re.sub(
            r"_+",
            "_",
            value,
        )

        return value.strip("_")

    def _generate_document_id(
        self,
        name: str,
    ) -> str:

        base_id = (
            self._slugify(name)
            or "document"
        )

        metadata = self._read_metadata()

        existing_ids = {
            item.get("id")
            for item in metadata["documents"]
            if isinstance(item, dict)
        }

        document_id = base_id

        counter = 2

        while document_id in existing_ids:

            document_id = (
                f"{base_id}_{counter}"
            )

            counter += 1

        return document_id

    # =================================================================
    # FORMAT
    # =================================================================

    @staticmethod
    def detect_format(
        file_name: str,
    ) -> str:

        suffix = (
            Path(file_name)
            .suffix
            .lower()
        )

        if suffix == ".txt":
            return "txt"

        if suffix == ".md":
            return "md"

        if suffix == ".pdf":
            return "pdf"

        if suffix == ".docx":
            return "docx"

        raise ValueError(
            "Unsupported knowledge document format. "
            "Supported formats: TXT, MD, PDF, DOCX."
        )

    # =================================================================
    # STREAMLIT UPLOAD
    # =================================================================

    def save_uploaded_file(
        self,
        uploaded_file,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:

        file_name = (
            uploaded_file.name
        )

        document_format = (
            self.detect_format(
                file_name
            )
        )

        document_name = (
            name.strip()
            if name
            else Path(
                file_name
            ).stem
        )

        document_id = (
            self._generate_document_id(
                document_name
            )
        )

        document_dir = (
            self.upload_dir
            / document_id
        )

        document_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        destination = (
            document_dir
            / file_name
        )

        try:

            with open(
                destination,
                "wb",
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )

        except Exception:

            shutil.rmtree(
                document_dir,
                ignore_errors=True,
            )

            raise

        return self.register_saved_file(
            document_id=document_id,
            name=document_name,
            file_path=destination,
            document_format=document_format,
        )

    # =================================================================
    # REGISTER EXISTING FILE
    # =================================================================

    def register_file(
        self,
        file_path,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:

        source = Path(
            file_path
        )

        if not source.exists():

            raise FileNotFoundError(
                f"Knowledge document not found: {source}"
            )

        document_format = (
            self.detect_format(
                source.name
            )
        )

        document_name = (
            name.strip()
            if name
            else source.stem
        )

        document_id = (
            self._generate_document_id(
                document_name
            )
        )

        document_dir = (
            self.upload_dir
            / document_id
        )

        document_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        destination = (
            document_dir
            / source.name
        )

        try:

            shutil.copy2(
                source,
                destination,
            )

        except Exception:

            shutil.rmtree(
                document_dir,
                ignore_errors=True,
            )

            raise

        return self.register_saved_file(
            document_id=document_id,
            name=document_name,
            file_path=destination,
            document_format=document_format,
        )

    # =================================================================
    # REGISTER STORED FILE
    # =================================================================

    def register_saved_file(
        self,
        document_id: str,
        name: str,
        file_path: Path,
        document_format: Optional[str] = None,
    ) -> Dict[str, Any]:

        file_path = Path(
            file_path
        )

        if document_format is None:

            document_format = (
                self.detect_format(
                    file_path.name
                )
            )

        metadata = (
            self._read_metadata()
        )

        now = (
            datetime.now()
            .isoformat()
        )

        document_metadata = {
            "id": document_id,
            "name": name,
            "filename": file_path.name,
            "path": str(
                file_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "format": document_format,
            "status": "stored",
            "indexed": False,
            "chunks": 0,
            "uploaded_at": now,
            "updated_at": now,
        }

        metadata["documents"].append(
            document_metadata
        )

        self._write_metadata(
            metadata
        )

        return document_metadata

    # =================================================================
    # LIST
    # =================================================================

    def list_documents(
        self,
    ) -> List[Dict[str, Any]]:

        metadata = (
            self._read_metadata()
        )

        return metadata[
            "documents"
        ]

    # =================================================================
    # GET
    # =================================================================

    def get_document(
        self,
        document_id: str,
    ) -> Optional[Dict[str, Any]]:

        for document in self.list_documents():

            if document.get("id") == document_id:

                return document

        return None

    # =================================================================
    # GET PATH
    # =================================================================

    def get_document_path(
        self,
        document_id: str,
    ) -> Path:

        document = (
            self.get_document(
                document_id
            )
        )

        if document is None:

            raise KeyError(
                f"Knowledge document not found: "
                f"{document_id}"
            )

        path = (
            PROJECT_ROOT
            / document["path"]
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Stored knowledge document not found: "
                f"{path}"
            )

        return path

    # =================================================================
    # UPDATE INDEX STATUS
    # =================================================================

    def mark_indexed(
        self,
        document_id: str,
        chunks: int = 0,
    ) -> bool:

        metadata = (
            self._read_metadata()
        )

        for document in metadata[
            "documents"
        ]:

            if document.get("id") == document_id:

                document["indexed"] = True

                document["chunks"] = int(
                    chunks
                )

                document["status"] = (
                    "indexed"
                )

                document["updated_at"] = (
                    datetime.now()
                    .isoformat()
                )

                self._write_metadata(
                    metadata
                )

                return True

        return False

    # =================================================================
    # DELETE
    # =================================================================

    def delete_document(
        self,
        document_id: str,
    ) -> bool:

        metadata = (
            self._read_metadata()
        )

        documents = metadata[
            "documents"
        ]

        target = None

        for document in documents:

            if document.get("id") == document_id:

                target = document
                break

        if target is None:

            return False

        document_dir = (
            self.upload_dir
            / document_id
        )

        if document_dir.exists():

            shutil.rmtree(
                document_dir
            )

        metadata["documents"] = [
            document
            for document in documents
            if document.get("id")
            != document_id
        ]

        self._write_metadata(
            metadata
        )

        return True

    # =================================================================
    # COUNT
    # =================================================================

    def count(self) -> int:

        return len(
            self.list_documents()
        )


# =====================================================================
# CLI TEST
# =====================================================================

if __name__ == "__main__":

    manager = KnowledgeManager()

    print()
    print("=" * 70)
    print("APMOA KNOWLEDGE MANAGER")
    print("=" * 70)

    print()
    print(
        "Storage directory:"
    )

    print(
        manager.upload_dir
    )

    print()

    documents = (
        manager.list_documents()
    )

    print(
        f"Registered documents: "
        f"{len(documents)}"
    )

    for document in documents:

        print()

        print(
            f"ID       : {document.get('id')}"
        )

        print(
            f"Name     : {document.get('name')}"
        )

        print(
            f"Format   : {document.get('format')}"
        )

        print(
            f"Indexed  : {document.get('indexed')}"
        )

        print(
            f"Chunks   : {document.get('chunks')}"
        )

        print(
            f"Status   : {document.get('status')}"
        )

    print()