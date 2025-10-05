import re
import io
import os
import shutil
import warnings

import PyPDF2
import pytesseract
from pytesseract import TesseractNotFoundError
from PIL import Image

# Optional: PDF → image conversion
try:
    from pdf2image import convert_from_bytes
    POPPLER_AVAILABLE = True
except ImportError:
    POPPLER_AVAILABLE = False
    warnings.warn("pdf2image/poppler not available. Scanned-PDF OCR will be limited.")

# Locate Tesseract
TESSERACT_AVAILABLE = False
for cand in [
    shutil.which("tesseract"),
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/usr/bin/tesseract",
]:
    if cand and os.path.exists(cand):
        pytesseract.pytesseract.tesseract_cmd = cand
        TESSERACT_AVAILABLE = True
        break
if not TESSERACT_AVAILABLE:
    warnings.warn("Tesseract not found. Image OCR disabled. Install via: brew install tesseract")

# ─────────────────────────────────────────────────────────────────────────────
# 1) Gender-specific reference ranges
NORMAL_RANGES = {
    "male": {
        "hemoglobin":  ((13.5, 17.5), "g/dL"),
        "rbc_count":   ((4.7, 6.1),   "mill/cumm"),
        "wbc_count":   ((4000, 11000),"cells/cumm"),
        "platelet_count": ((150000,450000),"cells/cumm"),
        "glucose":     ((70, 100),    "mg/dL"),
        "neutrophils": ((50, 62),     "%"),
        "lymphocytes": ((20, 40),     "%"),
        "eosinophils":((0, 6),       "%"),
        "monocytes":  ((0, 10),      "%"),
        "basophils":  ((0, 2),       "%"),
        "mcv":        ((83, 101),    "fL"),
        "mch":        ((27, 32),     "pg"),
        "mchc":       ((32.5,34.5),  "g/dL"),
        "pcv":        ((40, 50),     "%"),
        "rdw":        ((11.6,14.0),  "%")
    },
    "female": {
        "hemoglobin":  ((12.0, 15.5), "g/dL"),
        "rbc_count":   ((4.2, 5.4),   "mill/cumm"),
        "wbc_count":   ((4000,11000), "cells/cumm"),
        "platelet_count": ((150000,450000),"cells/cumm"),
        "glucose":     ((70, 100),    "mg/dL"),
        "neutrophils": ((50, 62),     "%"),
        "lymphocytes": ((20, 40),     "%"),
        "eosinophils":((0, 6),       "%"),
        "monocytes":  ((0, 10),      "%"),
        "basophils":  ((0, 2),       "%"),
        "mcv":        ((83, 101),    "fL"),
        "mch":        ((27, 32),     "pg"),
        "mchc":       ((32.5,34.5),  "g/dL"),
        "pcv":        ((36, 46),     "%"),
        "rdw":        ((11.6,14.0),  "%")
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# 2) Patterns to extract numeric values
patterns = {
    "hemoglobin":     r"(?:hemoglobin\s*KATEX_INLINE_OPENhbKATEX_INLINE_CLOSE|hb)[^\d\n]*?([\d\.]+)",
    "rbc_count":      r"(?:total\s*rbc\s*count)[^\d\n]*?([\d\.]+)",
    "wbc_count":      r"(?:total\s*wbc\s*count)[^\d\n]*?([\d\.]+)",
    "platelet_count": r"(?:platelet\s*count)[^\d\n]*?([\d\.]+)",
    "glucose":        r"(?:glucose|blood\s*sugar)[^\d\n]*?([\d\.]+)",
    "neutrophils":    r"(?:neutrophils)[^\d\n]*?([\d\.]+)",
    "lymphocytes":    r"(?:lymphocytes)[^\d\n]*?([\d\.]+)",
    "eosinophils":    r"(?:eosinophils)[^\d\n]*?([\d\.]+)",
    "monocytes":      r"(?:monocytes)[^\d\n]*?([\d\.]+)",
    "basophils":      r"(?:basophils)[^\d\n]*?([\d\.]+)",
    "mcv":            r"(?:mean\s*corpuscular\s*volume\s*KATEX_INLINE_OPENmcvKATEX_INLINE_CLOSE|mcv)[^\d\n]*?([\d\.]+)",
    "mch":            r"(?:mch)[^\d\n]*?([\d\.]+)",
    "mchc":           r"(?:mchc)[^\d\n]*?([\d\.]+)",
    "pcv":            r"(?:packed\s*cell\s*volume\s*KATEX_INLINE_OPENpcvKATEX_INLINE_CLOSE|pcv)[^\d\n]*?([\d\.]+)",
    "rdw":            r"(?:rdw)[^\d\n]*?([\d\.]+)",
    "age":            r'(?:age|patient\s*age|age:)\s*[:\-]?\s*(\d{1,3})'
}

# ─────────────────────────────────────────────────────────────────────────────
def detect_gender(text: str) -> str | None:
    """Return 'male', 'female', or None if not found."""
    t = text.lower()
    male_patterns = [r"\bmale\b", r"sex[:\s]*m\b", r"patient\s+mr\.?"]
    female_patterns = [r"\bfemale\b", r"sex[:\s]*f\b", r"patient\s+(mrs?|ms|miss)\.?"]

    for p in male_patterns:
        if re.search(p, t):
            return "male"
    for p in female_patterns:
        if re.search(p, t):
            return "female"
    return None

# ─────────────────────────────────────────────────────────────────────────────
def extract_metadata(text: str) -> dict:
    """
    Extracts:
      - patient_name
      - pathology_name
      - gender (via detect_gender)
      - age
    """
    meta = {"patient_name": "N/A", "pathology_name": "N/A", "gender": None, "age": "N/A"}
    lines = text.splitlines()

    # Patient Name
    for pat in [r"patient\s*[:\-]\s*([A-Z][\w\s\.]+)",
                r"name\s*[:\-]\s*([A-Z][\w\s\.]+)"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            meta["patient_name"] = m.group(1).strip().title()
            break

    # Pathology Lab
    t_lower = text.lower()
    for pat in [r"^([\w\s]+lab)", r"([\w\s]+pathology\s*lab)"]:
        m = re.search(pat, t_lower, re.IGNORECASE | re.MULTILINE)
        if m:
            name = m.group(1).strip()
            if len(name) > 5:
                meta["pathology_name"] = name.title()
                break

    # Gender
    meta["gender"] = detect_gender(text)
    if meta["gender"] is None:
        meta["gender"] = "Male"  # default

    # Age
    age_pat = r'(?:age|patient\s*age|age:)\s*[:\-]?\s*(\d{1,3})'
    age_match = re.search(age_pat, text, re.IGNORECASE)
    if age_match:
        age_str = age_match.group(1)
        try:
            age = int(age_str)
            if 1 <= age <= 120:  # Reasonable age range
                meta["age"] = age
        except ValueError:
            pass
        
    return meta

# ─────────────────────────────────────────────────────────────────────────────
def process_uploaded_pdf(file) -> str:
    """
    Reads file-like PDF and returns plain text.
    Tries:
      1) PyPDF2
      2) pdf2image → Tesseract OCR
      3) pdfplumber
      4) PIL → single-page OCR
    """
    data = file.read()

    # 1: PyPDF2 text
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        txt = ""
        for p in reader.pages:
            t = p.extract_text() or ""
            txt += t + "\n"
        if len(txt.strip()) > 50:
            return txt
    except Exception:
        pass

    # 2: pdf2image + Tesseract
    if POPPLER_AVAILABLE and TESSERACT_AVAILABLE:
        try:
            imgs = convert_from_bytes(data, dpi=300)
            txt = ""
            cfg = "--psm 6 --oem 3 -c tessedit_enable_doc_dict=0"
            for img in imgs:
                txt += pytesseract.image_to_string(img, lang="eng", config=cfg) + "\n"
            if len(txt.strip()) > 50:
                return txt
        except Exception:
            pass

    # 3: pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            txt = ""
            for p in pdf.pages:
                txt += (p.extract_text() or "") + "\n"
            if len(txt.strip()) > 50:
                return txt
    except Exception:
        pass

    # 4: Single-page PIL OCR
    if TESSERACT_AVAILABLE:
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            cfg = "--psm 6 --oem 3 -c tessedit_enable_doc_dict=0"
            return pytesseract.image_to_string(img, lang="eng", config=cfg)
        except Exception:
            pass

    return "Could not extract text from PDF. It may be scanned, password-protected, or corrupted."

# ─────────────────────────────────────────────────────────────────────────────
def extract_text_from_image(file) -> str:
    """Reads file-like image (PNG/JPG) and returns plain text via Tesseract OCR."""
    if not TESSERACT_AVAILABLE:
        return "Error: Tesseract OCR not installed."
    try:
        data = file.read() if hasattr(file, "read") else open(file, "rb").read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        cfg = "--psm 6 --oem 3 -c tessedit_enable_doc_dict=0"
        return pytesseract.image_to_string(img, lang="eng", config=cfg)
    except Exception as e:
        return f"Error processing image: {e}"

# ─────────────────────────────────────────────────────────────────────────────
def analyze_text(text: str, gender: str | None = None) -> dict:
    """
    1) Auto-detect gender if None
    2) Extract numeric values via regex
    3) Compare against NORMAL_RANGES
    4) Return dict with 'detected_gender' + each key → {value, unit, status, normal_range}
    """
    # 1) Gender
    if gender is None:
        gender = detect_gender(text) or "male"
    else:
        gender = gender.lower()
    gender = gender if gender in NORMAL_RANGES else "male"

    results = {"detected_gender": gender}
    nr = NORMAL_RANGES[gender]

    # 2) Pattern matching
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if not m:
            continue
        try:
            val = float(m.group(1))
        except:
            continue

        # 3) Normalize & compare
        if key in nr:
            lo, hi = nr[key][0]
            unit = nr[key][1]
            status = "Normal" if lo <= val <= hi else "Abnormal"
            results[key] = {
                "value": val,
                "unit": unit,
                "status": status,
                "normal_range": f"{lo} - {hi}"
            }
        else:
            results[key] = {"value": val, "unit": "N/A", "status": "No range", "normal_range": "N/A"}

    return results
