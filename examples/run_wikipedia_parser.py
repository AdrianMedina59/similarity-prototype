"""Example runner for Phase_3.wikipedia_parser

Usage:
    python examples/run_wikipedia_parser.py <wikipedia_url>

Prints first few paragraphs and their sentence splits.
"""
import sys
from pathlib import Path
import argparse

# Ensure project root is on path if run from examples/
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

try:
    # Prefer top-level parser if present
    from wikipedia_parser import parse_url_to_paragraph_sentences
    HAVE_TOP_LEVEL = True
except Exception:
    from wikipedia_parser import parse_url_to_paragraph_sentences
    HAVE_TOP_LEVEL = False


def main():
    parser = argparse.ArgumentParser(
        prog="run_wikipedia_parser.py",
        description="Fetch a Wikipedia URL and print article paragraphs and sentence splits."
    )
    parser.add_argument("url", help="Full Wikipedia article URL")
    parser.add_argument("--max", type=int, default=None,
                        help="Maximum number of paragraphs to parse (default: all)")
    parser.add_argument("--tfidf", action="store_true",
                        help="Run TF-IDF vectorization on all extracted sentences and print a compact summary")
    args = parser.parse_args()

    results = parse_url_to_paragraph_sentences(args.url, max_paragraphs=args.max)
    if args.tfidf:
        # Perform TF-IDF vectorization using the helper in the parser module
        # prefer top-level helper if available
        if HAVE_TOP_LEVEL:
            from wikipedia_parser import vectorize_paragraphs_tfidf
        else:
            from wikipedia_parser import vectorize_paragraphs_tfidf

        vec_result = vectorize_paragraphs_tfidf(results)
        total_sentences = len(vec_result["flat"]) 
        vocab_size = len(vec_result["vocabulary"])
        vec_len = len(vec_result["vectors"][0]) if total_sentences > 0 and len(vec_result["vectors"][0])>0 else 0
        print(f"Parsed {len(results)} paragraph(s) from {args.url}")
        print(f"Total sentences: {total_sentences}")
        print(f"Vocabulary size: {vocab_size}")
        print(f"Vector length: {vec_len}")
        # show sample of first 5 sentence -> vector non-zero counts
        print("\nSample sentence vector (first 5):")
        for i, item in enumerate(vec_result["flat"][:5], 1):
            vec = vec_result["vectors"][i-1]
            non_zero = sum(1 for v in vec if v != 0)
            print(f" {i}. Para {item['paragraph_index']} S{item['sentence_index']}: non-zero features={non_zero}")
        return

    print(f"Parsed {len(results)} paragraph(s) from {args.url}\n")
    for i, item in enumerate(results, 1):
        print(f"=== Paragraph {i} ===")
        # Print the paragraph (truncate long ones to keep output readable)
        print(item["paragraph"]) 
        print("\nSentences:")
        for j, s in enumerate(item["sentences"], 1):
            print(f" {j}. {s}")
        print()


if __name__ == "__main__":
    main()
