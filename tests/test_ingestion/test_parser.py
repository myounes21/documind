from pathlib import Path
import pytest
from ingestion.parser import parse_document

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
VALID_PDF = str(FIXTURES_DIR / "careerAi.pdf")


@pytest.fixture(scope="class")
def parsed_pdf():
    return parse_document(VALID_PDF)


class TestParseDocumentValidInput:

    def test_returns_list(self, parsed_pdf):
        assert isinstance(parsed_pdf, list)

    def test_returns_non_empty_list(self, parsed_pdf):
        assert len(parsed_pdf) > 0

    def test_elements_are_not_none(self, parsed_pdf):
        assert all(element is not None for element in parsed_pdf)


class TestParseDocumentInvalidInput:

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="file_path must not be empty"):
            parse_document("")

    def test_nonexistent_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_document(str(FIXTURES_DIR / "ghost.pdf"))

    def test_unsupported_file_type_raises_value_error(self, tmp_path):
        fake_file = tmp_path / "document.xyz"
        fake_file.write_text("some content")
        with pytest.raises(ValueError):
            parse_document(str(fake_file))


class TestParseDocumentEdgeCases:

    def test_whitespace_only_path_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_document("   ")

    def test_path_as_string_is_accepted(self, parsed_pdf):
        assert parsed_pdf is not None

    def test_absolute_path_works(self, parsed_pdf):
        assert len(parsed_pdf) > 0
