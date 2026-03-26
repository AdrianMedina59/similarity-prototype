import sys
from pathlib import Path

# Add parent directory to path so we can import wikipedia_parser
sys.path.insert(0, str(Path(__file__).parent.parent))

from wikipedia_parser import parse_url_to_paragraph_sentences

url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
results = parse_url_to_paragraph_sentences(url, max_paragraphs=5)

for item in results:
    print(f"Sentences: {len(item['sentences'])}")
    for s in item['sentences']:
        print(f"  - {s[:80]}")