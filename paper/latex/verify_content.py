import re
import sys
from pypdf import PdfReader

def dehyphenate(text):
    # Fix words split across lines with hyphens like 'eval-\nuator' or 'eval- uator'
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text) # potential space after hyphen in extraction
    return text

def verify():
    reader = PdfReader("main.pdf")
    num_pages = len(reader.pages)
    print(f"Total pages: {num_pages}")
    
    raw_pdf_text = ""
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        raw_pdf_text += f"\n--- Page {idx+1} ---\n" + text
        
    with open("extracted_pdf_text.txt", "w", encoding="utf-8") as f:
        f.write(raw_pdf_text)
        
    with open("../paper.md", "r", encoding="utf-8") as f:
        md_text = f.read()

    # Normalized text for search
    cleaned_pdf_text = re.sub(r'-\s*\n\s*', '', raw_pdf_text)
    cleaned_pdf_text_flat = re.sub(r'\s+', ' ', cleaned_pdf_text)

    # 1. Check for undefined citations
    unresolved_cites = re.findall(r"\[\?\]|\?\?", raw_pdf_text)
    print(f"Unresolved citations found: {len(unresolved_cites)}")
    assert len(unresolved_cites) == 0, "Found unresolved citations in PDF!"

    # 2. Check required hedges
    hedges = [
        "indicates",
        "is consistent with",
        "does not show",
        "at this sample size"
    ]
    print("\n--- Hedge checks ---")
    for hedge in hedges:
        in_md = hedge.lower() in md_text.lower()
        in_pdf = hedge.lower() in cleaned_pdf_text_flat.lower()
        print(f"Hedge '{hedge}': in MD = {in_md}, in PDF = {in_pdf}")
        assert in_pdf, f"Missing hedge '{hedge}' in PDF!"

    # 3. Check commit hashes
    commit_hashes = ["ce947d2", "a1fa9e5", "ddb20f0", "3445b67"]
    print("\n--- Commit hashes checks ---")
    for ch in commit_hashes:
        in_md = ch in md_text
        in_pdf = ch in cleaned_pdf_text_flat
        print(f"Commit hash '{ch}': in MD = {in_md}, in PDF = {in_pdf}")
        assert in_pdf, f"Missing commit hash '{ch}' in PDF!"

    # 4. Check Section 10.4 single annotator disclosure
    single_annotator_terms = [
        "single-annotator",
        "single evaluator",
        "permanent purge of synthetic second-annotator artifacts"
    ]
    print("\n--- Single annotator checks ---")
    for term in single_annotator_terms:
        # replace hyphen if needed
        norm_term = term.replace("-", "")
        in_pdf = norm_term.lower() in cleaned_pdf_text_flat.replace("-", "").lower()
        print(f"Term '{term}': in PDF = {in_pdf}")
        assert in_pdf, f"Missing term '{term}' in PDF!"

    # 5. Check key statistical numbers
    key_numbers = [
        "5,900+", "9,600", "0.86", "0.60", "2.50", "8.1",
        "56.67%", "60.00%", "81.11%", "80.6%", "25.0%",
        "+0.0831", "2.18", "-0.0665", "2.53", "0.1630",
        "-0.0129", "-0.0051",
        "25.00%", "53.68%", "36.76%", "86.76%", "80.88%", "90.44%",
        "52.08%", "76.56%", "2.05", "2.049", "8.068", "1.113", "9.766",
        "3.243", "4.480",
        "47 / 8", "70 / 2", "14 / 1", "61 / 8", "81 / 7", "56 / 9",
        "0.00289", "0.0807", "0.5811", "0.8167", "82.54%",
        "37.78%", "67.39%", "0.9981", "0.5610", "0.0057", "98.98%",
        "0.7041", "93.75%",
        "+0.0398", "+0.0200", "-0.0490", "6.350", "6.301",
        "6.312", "6.370", "6.357", "6.386", "6.390", "6.391", "6.392"
    ]
    print("\n--- Key number checks ---")
    missing_numbers = []
    # Normalize unicode minus sign \u2212 to ASCII -
    norm_pdf_all = re.sub(r'[^0-9a-zA-Z\+\-\.\%]', '', raw_pdf_text.replace('\u2212', '-'))
    for num in key_numbers:
        norm_num = re.sub(r'[^0-9a-zA-Z\+\-\.\%]', '', num.replace('\u2212', '-'))
        if norm_num not in norm_pdf_all:
            missing_numbers.append(num)
            print(f"WARNING: Number '{num}' not directly matched in normalized PDF text!")
        else:
            print(f"Number '{num}': MATCHED")
            
    print(f"\nTotal key numbers checked: {len(key_numbers)}, missing: {len(missing_numbers)}")
    assert len(missing_numbers) == 0, f"Missing numbers: {missing_numbers}"

    # 6. Check all 14 bibliography entries
    bib_authors = [
        "Chefer", "Chen", "Esser", "Gokhale", "Hertz", "Ho",
        "Huang", "Lin", "Oppenlaender", "Peebles", "Raffel",
        "Rombach", "Yang"
    ]
    print("\n--- Bibliography checks ---")
    for author in bib_authors:
        in_pdf = author in raw_pdf_text
        print(f"Author '{author}': in PDF = {in_pdf}")
        assert in_pdf, f"Missing author '{author}' in PDF bibliography!"

    print("\nALL INTEGRITY CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    verify()
