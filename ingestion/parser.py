from unstructured.partition.auto import partition
from unstructured.documents.elements import Element
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".html"}


def _is_file_supported(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS


def parse_document(file_path: str) -> list[Element]:
    """
    Parse a document file into a list of unstructured elements.

    Args:
        file_path: Absolute or relative path to the document file.

    Returns:
        A list of Element objects extracted from the document.

    Raises:
        ValueError: If file_path is empty or the file type is not supported.
        FileNotFoundError: If the file does not exist at the given path.
        RuntimeError: If the file is corrupt or fails during partitioning.
    """
    if not file_path:
        raise ValueError("file_path must not be empty.")

    if not _is_file_supported(file_path):
        raise ValueError(f"Unsupported file extension for '{file_path}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")

    try:
        elements = partition(filename=file_path)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to parse document '{file_path}': {e}")

    return elements
