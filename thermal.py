import fitz
import os
from collections import Counter

PDF = "input/thermal.pdf"

doc = fitz.open(PDF)

total = 0
per_page = {}

print("Scanning PDF images...\n")

for page_no, page in enumerate(doc, start=1):
    images = page.get_images(full=True)
    count = len(images)
    per_page[page_no] = count
    total += count

    print(f"Page {page_no}: {count} images")

print("\nTOTAL images in PDF:", total)
