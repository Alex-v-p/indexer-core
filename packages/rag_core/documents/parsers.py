from __future__ import annotations

from pathlib import Path
from typing import Protocol

from packages.rag_core.documents.models import ParsedDocument, ParsedPage

SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}
SUPPORTED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "application/octet-stream",  # browsers often send this for local files
}


class DocumentParser(Protocol):
    """Parser contract for source documents."""

    name: str
    version: str

    def parse(self, path: Path, *, filename: str | None = None, content_type: str | None = None) -> ParsedDocument:
        """Parse a stored file into page/logical-unit text."""


class UnsupportedDocumentTypeError(ValueError):
    """Raised when no parser is registered for a file."""


class TextDocumentParser:
    name = "text"
    version = "0.1.0"

    def parse(self, path: Path, *, filename: str | None = None, content_type: str | None = None) -> ParsedDocument:
        text = _read_text_with_fallback(path)
        title = _title_from_filename(filename or path.name)
        return ParsedDocument(
            title=title,
            pages=[ParsedPage(page_number=None, text=text, metadata={"source_unit": "text"})],
            parser_name=self.name,
            parser_version=self.version,
            metadata={"content_type": content_type, "source_filename": filename or path.name},
        )


class MarkdownDocumentParser(TextDocumentParser):
    name = "markdown"
    version = "0.1.0"


class PdfDocumentParser:
    name = "pdf"
    version = "0.1.0"

    def parse(self, path: Path, *, filename: str | None = None, content_type: str | None = None) -> ParsedDocument:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - dependency should be installed by requirements.txt
            raise RuntimeError("PDF ingestion requires the pypdf package.") from exc

        reader = PdfReader(str(path))
        pages: list[ParsedPage] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(
                ParsedPage(
                    page_number=index,
                    text=text,
                    metadata={"source_unit": "page", "page_number": index},
                ),
            )

        title = _title_from_filename(filename or path.name)
        return ParsedDocument(
            title=title,
            pages=pages,
            parser_name=self.name,
            parser_version=self.version,
            metadata={
                "content_type": content_type,
                "source_filename": filename or path.name,
                "page_count": len(pages),
            },
        )


def parse_document(path: Path, *, filename: str | None = None, content_type: str | None = None) -> ParsedDocument:
    """Parse a supported source document."""

    parser = get_parser_for_document(filename or path.name, content_type=content_type)
    return parser.parse(path, filename=filename, content_type=content_type)


def get_parser_for_document(filename: str, *, content_type: str | None = None) -> DocumentParser:
    suffix = Path(filename).suffix.lower()
    normalized_content_type = (content_type or "").split(";")[0].strip().lower()

    if suffix == ".pdf" or normalized_content_type == "application/pdf":
        return PdfDocumentParser()
    if suffix in {".md", ".markdown"} or normalized_content_type in {"text/markdown", "text/x-markdown"}:
        return MarkdownDocumentParser()
    if suffix == ".txt" or normalized_content_type == "text/plain":
        return TextDocumentParser()

    raise UnsupportedDocumentTypeError(
        "Unsupported document type. Supported formats are PDF, text, and markdown.",
    )


def is_supported_document(filename: str, *, content_type: str | None = None) -> bool:
    suffix = Path(filename).suffix.lower()
    normalized_content_type = (content_type or "").split(";")[0].strip().lower()
    return suffix in SUPPORTED_DOCUMENT_EXTENSIONS or normalized_content_type in SUPPORTED_CONTENT_TYPES


def _read_text_with_fallback(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _title_from_filename(filename: str) -> str:
    stem = Path(filename).stem.strip()
    return stem or "Untitled document"
