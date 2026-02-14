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

MAX_IMAGES_PER_AREA = 2

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
    os.makedirs("build", exist_ok=True)
    os.makedirs(INS_IMG_DIR, exist_ok=True)


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


# =======================
# TEXT EXTRACTION
# =======================
def extract_text(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            t = page.extract_text()
            if t:
                pages.append({
                    "page": i,
                    "text": clean(t)
                })
    return pages


# =======================
# IMPACTED AREAS
# =======================
def extract_impacted_areas(pages):
    for p in pages:
        if "Impacted Areas/Rooms" in p["text"]:
            found = []
            for area in ALL_AREAS:
                if area.lower() in p["text"].lower():
                    found.append(area)
            return found
    return ALL_AREAS


# =======================
# IMAGE EXTRACTION (REFERENCE ONLY)
# =======================
def extract_inspection_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []

    for page_no, page in enumerate(doc, start=1):
        count = 0
        for img in page.get_images(full=True):
            if count >= MAX_IMAGES_PER_AREA:
                break

            base = doc.extract_image(img[0])
            w, h = base["width"], base["height"]

            if w < 300 or h < 300:
                continue

            name = f"page{page_no}_img{count}.{base['ext']}"
            path = os.path.join(INS_IMG_DIR, name)

            with open(path, "wb") as f:
                f.write(base["image"])

            images.append({
                "page": page_no,
                "path": path
            })
            count += 1

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


def chunk_global(pages, source):
    return [{
        "area": "GLOBAL",
        "page": p["page"],
        "source": source,
        "text": p["text"]
    } for p in pages]


# =======================
# IMAGE ↔ AREA MAPPING
# =======================
def map_images_to_areas(images, area_chunks):
    mapping = {}

    for ch in area_chunks:
        area = ch["area"]
        page = ch["page"]

        for img in images:
            if img["page"] == page:
                mapping.setdefault(area, []).append(img)

    # limit images per area
    for area in mapping:
        mapping[area] = mapping[area][:MAX_IMAGES_PER_AREA]

    return mapping


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
def generate_ddr(embedder, index, docs, impacted_areas, area_chunks):
    sections = {}

    sections["summary"] = llm(f"""
Generate Property Issue Summary.

Rules:
- Use ONLY provided context
- No assumptions

Context:
{json.dumps(retrieve(embedder, index, docs, "overall property issues"), indent=2)}
""")

    sections["root"] = llm(f"""
Generate Probable Root Cause.

Rules:
- Supported by text only
- If unclear, say "Not Available"

Context:
{json.dumps(retrieve(embedder, index, docs, "root cause leakage"), indent=2)}
""")

    sections["severity"] = llm(f"""
Assess severity (Low / Medium / High) with reasoning.

Context:
{json.dumps(retrieve(embedder, index, docs, "severity extent damage"), indent=2)}
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

    # Area-wise observations (TEXT ONLY)
    area_text = {}
    for area in impacted_areas:
        ctx = [c for c in area_chunks if c["area"] == area]
        if not ctx:
            continue

        text = llm(f"""
Write observations for {area}.

Rules:
- Describe observations ONLY
- Do NOT mention photos or images
- Do NOT add placeholders

Context:
{json.dumps(ctx, indent=2)}
""")

        area_text[area] = text.strip()

    sections["areas"] = area_text
    return sections


# =======================
# MAIN
# =======================
def main():
    ensure_dirs()

    inspection_text = extract_text(INSPECTION_PDF)
    thermal_text = extract_text(THERMAL_PDF)

    impacted_areas = extract_impacted_areas(inspection_text)

    inspection_images = extract_inspection_images(INSPECTION_PDF)

    area_chunks = (
        chunk_by_area(inspection_text, "inspection", impacted_areas) +
        chunk_by_area(thermal_text, "thermal", impacted_areas)
    )

    global_docs = (
        chunk_global(inspection_text, "inspection") +
        chunk_global(thermal_text, "thermal")
    )

    docs = global_docs + area_chunks

    embedder, index = build_index(docs)
    ddr = generate_ddr(embedder, index, docs, impacted_areas, area_chunks)

    area_image_map = map_images_to_areas(inspection_images, area_chunks)

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# Detailed Diagnostic Report\n\n")

        f.write("## 1. Property Issue Summary\n")
        f.write(ddr["summary"] + "\n\n")

        f.write("## 2. Area-wise Observations\n")
        for area in impacted_areas:
            if area not in ddr["areas"]:
                continue

            f.write(f"### {area}\n")
            f.write(ddr["areas"][area] + "\n")

            imgs = area_image_map.get(area, [])
            if imgs:
                f.write("\n**Supporting Visual Evidence**\n")
                for img in imgs:
                    rel = os.path.relpath(img["path"], start="build")
                    rel = rel.replace("\\", "/")  # WINDOWS FIX
                    f.write(f"\n![]({rel})\n")
            f.write("\n")

        f.write("## 3. Probable Root Cause\n")
        f.write(ddr["root"] + "\n\n")

        f.write("## 4. Severity Assessment\n")
        f.write(ddr["severity"] + "\n\n")

        f.write("## 5. Recommended Actions\n")
        f.write(ddr["actions"] + "\n\n")

        f.write("## 6. Missing or Unclear Information\n")
        f.write("- Plumbing layout: Not Available\n")
        f.write("- Moisture meter readings: Not Available\n")
        f.write("- Active leakage confirmation: Not Available\n")

    print("DDR generated successfully:", OUT_MD)


if __name__ == "__main__":
    main()
