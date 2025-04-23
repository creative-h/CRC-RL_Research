import sys
from pathlib import Path

try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not found. Please install it in your venv: pip install PyPDF2")
    sys.exit(1)

def extract_text(pdf_path, output_path=None):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        return
    reader = PyPDF2.PdfReader(str(pdf_path))
    all_text = ""
    for page in reader.pages:
        all_text += page.extract_text() or ""
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(all_text)
    else:
        print(all_text)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract text from a PDF file.")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output", help="Output text file (optional)")
    args = parser.parse_args()
    extract_text(args.pdf_path, args.output)
