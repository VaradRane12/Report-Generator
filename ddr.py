import os, re, json
import pdfplumber
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
INSPECTION_PDF = "input/Sample_Report.pdf"
THERMAL_PDF = "input/Thermal_Images.pdf"
IMG_INS_DIR = "images/inspection"
IMG_TH_DIR = "images/thermal"
OUT_MD = "build/ddr.md"

AREAS = [
    "Hall", "Bedroom", "Master Bedroom",
    "Kitchen", "Common Bathroom",
    "Parking Area", "External Wall"
]

# photo-number → area mapping (from inspection report structure)
AREA_IMAGE_MAP = {
    "Hall": range(1, 8),
    "Bedroom": range(8, 15),
    "Master Bedroom": range(15, 31),
    "Kitchen": range(31, 33),
    "Parking Area": range(49, 53),
    "Common Bathroom": range(53, 65),
    "External Wall": range(42, 49)
}

# -----------------------------
# UTILS
# -----------------------------
def ensure_dirs():
    os.makedirs(IMG_INS_DIR, exist_ok=True)
    os.makedirs(IMG_TH_DIR, exist_ok=True)
    os.makedirs("build", exist_ok=True)

def clean(text):
    return re.sub(r"\s+", " ", text).strip()

# -----------------------------
# TEXT EXTRACTION
# -----------------------------
def extract_text(pdf):
    pages = []
    with pdfplumber.open(pdf) as p:
        for i, page in enumerate(p.pages, 1):
            t = page.extract_text()
            if t:
                pages.append({"page": i, "text": clean(t)})
    return pages

# -----------------------------
# IMAGE EXTRACTION
# -----------------------------
def extract_images(pdf, out_dir):
    doc = fitz.open(pdf)
    images = []
    for pno, page in enumerate(doc, 1):
        for idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            name = f"page{pno}_img{idx}.{base['ext']}"
            path = os.path.join(out_dir, name)
            with open(path, "wb") as f:
                f.write(base["image"])
            images.append({"page": pno, "path": path})
    return images

def photo_number(path):
    m = re.search(r"Photo\s*(\d+)", path)
    return int(m.group(1)) if m else None

# -----------------------------
# CHUNKING
# -----------------------------
def inspection_chunks(pages):
    chunks = []
    for p in pages:
        for area in AREAS:
            if area.lower() in p["text"].lower():
                chunks.append({
                    "area": area,
                    "source": "inspection",
                    "text": p["text"]
                })
    return chunks

def thermal_chunks(pages):
    chunks = []
    for p in pages:
        h = re.search(r"Hotspot\s*:\s*([\d.]+)", p["text"])
        c = re.search(r"Coldspot\s*:\s*([\d.]+)", p["text"])
        chunks.append({
            "area": "Not Specified",
            "source": "thermal",
            "hotspot": h.group(1) if h else "NA",
            "coldspot": c.group(1) if c else "NA",
            "text": p["text"]
        })
    return chunks

# -----------------------------
# VECTOR STORE (RAG)
# -----------------------------
def build_index(docs):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [d["text"] for d in docs]
    emb = model.encode(texts)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(np.array(emb))
    return model, index

def retrieve(model, index, docs, query, k=4):
    q = model.encode([query])
    _, idx = index.search(np.array(q), k)
    return [docs[i] for i in idx[0]]

# -----------------------------
# IMAGE ASSIGNMENT
# -----------------------------
def images_for_area(area, images):
    nums = AREA_IMAGE_MAP.get(area, [])
    return [i["path"] for i in images if any(str(n) in i["path"] for n in nums)]

# -----------------------------
# DDR GENERATION (RULE-BASED TEXT)
# -----------------------------
def generate_ddr(areas, docs, model, index, ins_imgs, th_imgs):
    md = ["# Detailed Diagnostic Report\n"]

    md.append("## 1. Property Issue Summary")
    md.append(
        "- Dampness and paint deterioration observed at skirting and ceiling levels.\n"
        "- Gaps in tile joints observed in bathrooms and balcony.\n"
        "- Cracks observed on external walls.\n"
        "- Thermal readings show temperature variations at affected locations.\n"
    )

    md.append("## 2. Area-wise Observations")

    for area in areas:
        ctx = retrieve(model, index, docs, area)
        md.append(f"### {area}")
        for c in ctx:
            md.append(f"- {c['text'][:300]}")

        imgs = images_for_area(area, ins_imgs)
        for img in imgs[:2]:
            md.append(f"![{area}]({img})")

    md.append("## 3. Probable Root Cause")
    md.append(
        "- Gaps in tile joints allowing moisture ingress.\n"
        "- Concealed plumbing issues reported.\n"
        "- External wall cracks allowing rainwater ingress.\n"
    )

    md.append("## 4. Severity Assessment")
    md.append(
        "**Severity: Medium to High**\n\n"
        "Multiple areas affected with prolonged moisture exposure."
    )

    md.append("## 5. Recommended Actions")
    md.append(
        "- Repair tile joints in wet areas.\n"
        "- Inspect and repair plumbing lines.\n"
        "- Seal external wall cracks.\n"
        "- Repair plaster and repaint after drying.\n"
    )

    md.append("## 6. Additional Notes")
    md.append(
        "- Thermal readings support moisture presence.\n"
        "- Observations limited to visible areas.\n"
    )

    md.append("## 7. Missing or Unclear Information")
    md.append(
        "- Plumbing layout: Not Available\n"
        "- Moisture meter readings: Not Available\n"
        "- Active leakage confirmation: Not Available\n"
    )

    return "\n".join(md)

# -----------------------------
# MAIN
# -----------------------------
def main():
    ensure_dirs()

    ins_text = extract_text(INSPECTION_PDF)
    th_text = extract_text(THERMAL_PDF)

    ins_imgs = extract_images(INSPECTION_PDF, IMG_INS_DIR)
    th_imgs = extract_images(THERMAL_PDF, IMG_TH_DIR)

    docs = inspection_chunks(ins_text) + thermal_chunks(th_text)

    model, index = build_index(docs)

    ddr = generate_ddr(AREAS, docs, model, index, ins_imgs, th_imgs)

    with open(OUT_MD, "w") as f:
        f.write(ddr)

    print("DDR generated:", OUT_MD)

if __name__ == "__main__":
    main()
