import os
import re
import json
import subprocess

import pdfplumber
import fitz
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

ALL_AREAS = [
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
# TEXT EXTRACTION
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
# IMPACTED AREAS
# =======================
def extract_impacted_areas(pages):
    for p in pages:
        if "Impacted Areas/Rooms" in p["text"]:
            parts = re.split(r",|\n", p["text"])
            areas = []
            for part in parts:
                part = part.strip()
                if part in ALL_AREAS:
                    areas.append(part)
            return list(dict.fromkeys(areas))
    return []


# =======================
# IMAGE EXTRACTION
# =======================
def extract_inspection_images(pdf_path, out_dir):
    doc = fitz.open(pdf_path)
    images = []

    for page_no, page in enumerate(doc, start=1):
        for idx, img in enumerate(page.get_images(full=True)):
            base = doc.extract_image(img[0])
            w, h = base["width"], base["height"]
            area = w * h
            ratio = w / h if h else 0

            if w < 300 or h < 300:
                continue
            if area < 150_000:
                continue
            if 0.85 < ratio < 1.15 and area < 400_000:
                continue

            name = f"page{page_no}_img{idx}.{base['ext']}"
            path = os.path.join(out_dir, name)

            with open(path, "wb") as f:
                f.write(base["image"])

            images.append({
                "page": page_no,
                "path": path
            })

    return images


# =======================
# CHUNKING
# =======================
def chunk_by_area(pages, source, areas):
    chunks = []
    for p in pages:
        for area in areas:
            if area.lower() in p["text"].lower():
                chunks.append({
                    "area": area,
                    "page": p["page"],
                    "source": source,
                    "text": p["text"]
                })
    return chunks


# =======================
# IMAGE ↔ AREA LINKING
# =======================
def map_images_to_areas(images, chunks):
    area_images = {a["area"]: [] for a in chunks}

    for chunk in chunks:
        area = chunk["area"]
        page = chunk["page"]

        for img in images:
            if img["page"] == page:
                area_images.setdefault(area, []).append(img)

    return area_images


# =======================
# RAG
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
# LLM
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
def generate_ddr(embedder, index, docs, impacted_areas, area_images):
    sections = {}

    sections["summary"] = llm(f"""
Generate Property Issue Summary.
Use ONLY provided context.

Context:
{json.dumps(retrieve(embedder, index, docs, "overall property issues"), indent=2)}
""")

    area_text = ""
    for area in impacted_areas:
        ctx = retrieve(embedder, index, docs, area)
        area_text += f"\n### {area}\n"

        area_text += llm(f"""
Write observations for {area}.
Grounded only.

Context:
{json.dumps(ctx, indent=2)}
""")

        imgs = area_images.get(area, [])
        if imgs:
            area_text += "\n**Supporting Visual Evidence:**\n"
            for img in imgs:
                rel_path = os.path.relpath(img["path"], start="build")
                area_text += f"\n![Inspection image – Page {img['page']}]({rel_path})\n"


    sections["areas"] = area_text

    sections["root"] = llm(f"""
Generate Probable Root Cause.
If unclear, say "Not Available".

Context:
{json.dumps(retrieve(embedder, index, docs, "root cause"), indent=2)}
""")

    sections["severity"] = llm(f"""
Assess severity with reasoning.

Context:
{json.dumps(retrieve(embedder, index, docs, "severity extent damage"), indent=2)}
""")

    sections["actions"] = llm(f"""
Generate Recommended Actions.

Context:
{json.dumps(retrieve(embedder, index, docs, "recommended actions"), indent=2)}
""")

    return sections


# =======================
# MAIN
# =======================
def main():
    ensure_dirs()

    inspection_text = extract_text(INSPECTION_PDF)
    thermal_text = extract_text(THERMAL_PDF)

    impacted_areas = extract_impacted_areas(inspection_text)

    inspection_images = extract_inspection_images(
        INSPECTION_PDF, INS_IMG_DIR
    )

    docs = (
        chunk_by_area(inspection_text, "inspection", impacted_areas) +
        chunk_by_area(thermal_text, "thermal", impacted_areas)
    )

    area_images = map_images_to_areas(inspection_images, docs)

    embedder, index = build_index(docs)
    ddr = generate_ddr(embedder, index, docs, impacted_areas, area_images)

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
Inspection images are included as supporting visual evidence only.

## 7. Missing or Unclear Information
- Plumbing layout: Not Available
- Moisture meter readings: Not Available
- Active leakage confirmation: Not Available

## Appendix A: Inspection Photographs
Inspection photographs are provided separately in the project folder and referenced by page number above.
""")

    print("DDR generated successfully:", OUT_MD)


if __name__ == "__main__":
    main()
