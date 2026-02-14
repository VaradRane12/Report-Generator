## Requirements & What Is Needed

### Required Inputs
- `inspection.pdf` – site inspection report  
- `thermal.pdf` – thermal inspection report  

### Required Software
- Python **3.9 – 3.11**
- Ollama (local LLM) with `llama3`
- Pandoc
- wkhtmltopdf (for PDF generation)

### Required Python Packages
- pdfplumber
- pymupdf
- sentence-transformers
- faiss-cpu
- numpy

### What the System Produces
- `ddr.md` – structured Detailed Diagnostic Report
- `ddr.pdf` – final client-ready report with images

### Key Constraints
- No facts are invented
- Missing information is marked **Not Available**
- Images are used only as visual references
