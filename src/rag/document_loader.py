import re
import uuid
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Dict

try:
    import pypdf
except ImportError:
    pypdf = None

class DocumentLoader:
    """
    Loads and chunks knowledge documents from the APMOA knowledge directory.
    """

    def __init__(self, knowledge_dir: str = "data/knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

    def load_documents(self) -> List[Dict[str, str]]:
        """
        Load, parse, clean, and chunk documents.
        Returns a list of chunks, each retaining metadata.
        """
        chunks = []
        for file_path in sorted(self.knowledge_dir.glob("*")):
            if file_path.is_dir():
                continue

            ext = file_path.suffix.lower()
            text = ""

            if ext in {".txt", ".md"}:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            elif ext == ".pdf":
                if pypdf:
                    try:
                        reader = pypdf.PdfReader(str(file_path))
                        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
                    except Exception as e:
                        print(f"Error reading PDF {file_path.name}: {e}")
                else:
                    print(f"pypdf not installed. Skipping {file_path.name}")
            elif ext == ".docx":
                try:
                    with zipfile.ZipFile(str(file_path)) as docx:
                        xml_content = docx.read('word/document.xml')
                        tree = ET.fromstring(xml_content)
                        namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                        text = "\n".join(node.text for node in tree.findall('.//w:t', namespaces) if node.text)
                except Exception as e:
                    print(f"Error reading DOCX {file_path.name}: {e}")
            else:
                continue

            text = self._clean_text(text)
            if not text:
                continue

            file_chunks = self._chunk_text(text)
            created_at = datetime.now().isoformat()

            for i, chunk_text in enumerate(file_chunks):
                chunks.append({
                    "document_name": file_path.stem,
                    "document_type": ext.lstrip('.').upper(),
                    "chunk_id": f"{file_path.name}-chunk-{i}",
                    "source": file_path.name,
                    "created_at": created_at,
                    "text": chunk_text
                })

        return chunks

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += chunk_size - overlap
        return chunks
