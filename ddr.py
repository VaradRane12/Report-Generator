import os
import re
import json
import subprocess

import pdfplumber
import fitz  # pymupdf
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# =======================
# CONFIG
# =======================
INSPECTION_PDF = "input/inspection.pdf"
THERMAL_PDF = "input/thermal.pdf"

INS_IMG_DIR = "images/inspection"
OUT_MD = "build/ddr.md"

AREAS = [
    "Hall",
    "Bedroom",
    "Master Bedroom",
    "Kitchen",
    "Common Bathroom",
    "Parking Area",
    "External Wall",
    "Balcony",
    "Terrace"
]


# =======================
# UTILS
# =======================
def ensure_dirs():
    for d in [INS_IMG_DIR, "build"]:
        os.makedirs(d, exist_ok=True)


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


# =======================
# PAGE NUMBER MAPPING
# =======================
def extract_visible_page_numbers(pdf_path):
    mapping = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            m = re.search(r"Page\s*(\d+)", text)
            mapping[i] = int(m.group(1)) if m else i
    return mapping


# =======================
# TEXT EXTRACTION (USED FOR BOTH PDFs)
# =======================
def extract_text(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                pages.append({
                    "page": i,
                    "text": clean(text)
                })
    return pages


# =======================
# INSPECTION IMAGE EXTRACTION ONLY
# =======================
def extract_inspection_images(pdf_path, out_dir):
    doc = fitz.open(pdf_path)
    page_map = extract_visible_page_numbers(pdf_path)
    images = []

    for pdf_page, page in enumerate(doc, start=1):
        visible_page = page_map.get(pdf_page, pdf_page)

        for idx, img in enumerate(page.get_images(full=True)):
            base = doc.extract_image(img[0])
            w, h = base["width"], base["height"]
            area = w * h
            ratio = w / h if h else 0

            # Filters
            if w < 300 or h < 300:
                continue
            if area < 150_000:
                continue
            if 0.85 < ratio < 1.15 and area < 400_000:
                continue

            name = f"page{visible_page}_img{idx}.{base['ext']}"
            path = os.path.join(out_dir, name)

            with open(path, "wb") as f:
                f.write(base["image"])

            images.append({
                "page": visible_page,
                "path": path
            })

    return images


# =======================
# CHUNKING
# =======================
def chunk_by_area(pages, source):
    chunks = []
    for p in pages:
        for area in AREAS:
            if area.lower() in p["text"].lower():
                chunks.append({
                    "area": area,
                    "source": source,
                    "text": p["text"]
                })
    return chunks


# =======================
# RAG (FAISS)
# =======================
def build_index(docs):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode([d["text"] for d in docs])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return embedder, index


def retrieve(embedder, index, docs, query, k=6):
    q = embedder.encode([query])
    _, idx = index.search(np.array(q), k)
    return [docs[i] for i in idx[0]]


# =======================
# LLM (OLLAMA)
# =======================
def llm(prompt):
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()


# =======================
# DDR GENERATION
# =======================
def generate_ddr(embedder, index, docs):
    sections = {}

    sections["summary"] = llm(f"""
Generate Property Issue Summary.

Rules:
- Use ONLY provided context
- Do NOT assume causes

Context:
{json.dumps(retrieve(embedder, index, docs, "overall property issues"), indent=2)}
""")

    area_text = ""
    for area in AREAS:
        ctx = retrieve(embedder, index, docs, f"{area} dampness leakage thermal")
        if not ctx:
            continue

        area_text += f"\n### {area}\n"
        area_text += llm(f"""
Write observations for {area}.

Rules:
- Use inspection observations
- Use thermal data ONLY if explicitly mentioned in text
- Do NOT describe thermal images

Context:
{json.dumps(ctx, indent=2)}
""")

    sections["areas"] = area_text

    sections["root"] = llm(f"""
Generate Probable Root Cause.

Rules:
- Only mention causes supported by text
- If unclear, write "Not Available"

Context:
{json.dumps(retrieve(embedder, index, docs, "root cause leakage"), indent=2)}
""")

    sections["severity"] = llm(f"""
Assess severity (Low / Medium / High) with reasoning.

Context:
{json.dumps(retrieve(embedder, index, docs, "extent damage severity"), indent=2)}
""")

    sections["actions"] = llm(f"""
Generate Recommended Actions.

Rules:
- Conservative
- No guarantees
- No product names

Context:
{json.dumps(retrieve(embedder, index, docs, "recommended repairs"), indent=2)}
""")

    return sections


# =======================
# MAIN
# =======================
def main():
    ensure_dirs()

    inspection_text = extract_text(INSPECTION_PDF)
    thermal_text = extract_text(THERMAL_PDF)

    extract_inspection_images(INSPECTION_PDF, INS_IMG_DIR)

    docs = (
        chunk_by_area(inspection_text, "inspection") +
        chunk_by_area(thermal_text, "thermal")
    )

    embedder, index = build_index(docs)
    ddr = generate_ddr(embedder, index, docs)

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(f"""
# Detailed Diagnostic Report

## 1. Property Issue Summary
{ddr['summary']}

## 2. Area-wise Observations
{ddr['areas']}

## 3. Probable Root Cause
{ddr['root']}

## 4. Severity Assessment
{ddr['severity']}

## 5. Recommended Actions
{ddr['actions']}

## 6. Additional Notes
Observations are based on accessible areas only. Thermal analysis is derived from temperature readings provided in the report.

## 7. Missing or Unclear Information
- Plumbing layout: Not Available
- Moisture meter readings: Not Available
- Active leakage confirmation: Not Available
""")

    print("DDR generated successfully:", OUT_MD)


if __name__ == "__main__":
    main()
