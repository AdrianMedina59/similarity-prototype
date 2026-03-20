import pytest
try:
    from wikipedia_parser import parse_url_to_paragraph_sentences
except Exception:
    from Phase_3.wikipedia_parser import parse_url_to_paragraph_sentences


@pytest.mark.network
def test_parse_nlp_wikipedia_first_paragraph():
    url = "https://en.wikipedia.org/wiki/Natural_language_processing"
    results = parse_url_to_paragraph_sentences(url, max_paragraphs=1)
    assert isinstance(results, list)
    assert len(results) == 1
    item = results[0]
    assert "paragraph" in item and "sentences" in item
    assert len(item["paragraph"]) > 30
    assert isinstance(item["sentences"], list)
    # We expect at least 1 sentence
    assert len(item["sentences"]) >= 1
