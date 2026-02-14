from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import subprocess
import os
import torch

# -----------------------------
# Load BLIP (DIRECT, STABLE)
# -----------------------------
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model.eval()  # inference only


# -----------------------------
# Stage 1: Raw image caption
# -----------------------------
def raw_caption(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=40
        )

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


# -----------------------------
# Stage 2: DDR-safe refinement (TEXT ONLY)
# -----------------------------
def refine_caption(raw_text):
    prompt = f"""
You are refining an inspection photo description.

Rules:
- Describe ONLY visible wall, ceiling, or floor surface conditions
- Mention stains, discoloration, cracks, peeling paint if visible
- Do NOT guess causes
- Do NOT mention moisture source
- Do NOT mention severity
- If nothing relevant is visible, say "No visible surface damage noted"

Raw caption:
"{raw_text}"

Refined description:
"""

    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        text=True,
        capture_output=True
    )

    return result.stdout.strip()


# -----------------------------
# TEST INPUT
# -----------------------------
inspection_images = [
    {
        "page": 3,
        "path": "images/inspection/page3_img1.jpeg"
    },
    {
        "page": 3,
        "path": "images/inspection/page3_img2.jpeg"
    }
]


# -----------------------------
# RUN TEST
# -----------------------------
print("\nImage descriptions (FINAL, STABLE):\n")

for img in inspection_images:
    base = raw_caption(img["path"])
    refined = refine_caption(base)

    print(f"Page {img['page']} | {img['path']}")
    print(f"Raw caption: {base}")
    print(f"Refined description: {refined}\n")
