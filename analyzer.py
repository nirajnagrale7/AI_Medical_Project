# analyzer.py (Robust Version with Fallback)

import re
import os
import shutil
import warnings
import PyPDF2
import pytesseract
from pytesseract import TesseractNotFoundError
from PIL import Image
import io

# Try to import pdf2image, but handle if poppler is not available
try:
    from pdf2image import convert_from_bytes
    POPPLER_AVAILABLE = True
except ImportError:
    POPPLER_AVAILABLE = False
    print("Warning: Poppler not available. OCR fallback will be limited.")

# Try to locate Tesseract binary and configure pytesseract
TESSERACT_AVAILABLE = False
try:
    tesseract_candidates = [
        shutil.which('tesseract'),
        '/opt/homebrew/bin/tesseract',  # macOS (Apple Silicon) Homebrew
        '/usr/local/bin/tesseract',     # macOS (Intel) Homebrew
        '/usr/bin/tesseract',           # Linux
        'C\\\\Program Files\\\	esseract-ocr\\\	esseract.exe',  # Windows (common)
        'C:\\\\Program Files\\\\Tesseract-OCR\\\\tesseract.exe'  # Windows (common alternative)
    ]
    for cand in tesseract_candidates:
        if cand and os.path.exists(cand):
            pytesseract.pytesseract.tesseract_cmd = cand
            TESSERACT_AVAILABLE = True
            break
    if not TESSERACT_AVAILABLE:
        warnings.warn("Tesseract not found. OCR features will be disabled. Install via 'brew install tesseract' on macOS.")
except Exception:
    warnings.warn("Unable to verify Tesseract OCR installation; OCR features may not work.")

# --- NORMAL_RANGES ---
NORMAL_RANGES = {
    'hemoglobin': {'range': (13.5, 17.5), 'unit': 'g/dL'},
    'wbc_count': {'range': (4.5, 11.0), 'unit': '10^9/L'},
    'platelet_count': {'range': (150, 450), 'unit': '10^9/L'},
    'glucose': {'range': (70, 100), 'unit': 'mg/dL'}
}

# --- IMPROVED PATTERNS ---
patterns = {
    'hemoglobin': r'(?:[Hh]emoglobin|[Hh][Gg][Bb])[^\d\n]*([+-]?\d+(?:\.\d+)?)',
    'wbc_count': r'(?:[Ww][Bb][Cc]|[Ww]hite\s*[Bb]lood\s*[Cc]ell)[^\d\n]*([+-]?\d+(?:\.\d+)?)',
    'platelet_count': r'(?:[Pp]latelet|[Pp][Ll][Tt])[^\d\n]*([+-]?\d+(?:\.\d+)?)',
    'glucose': r'(?:[Gg]lucose|[Bb]lood\s*[Ss]ugar)[^\d\n]*([+-]?\d+(?:\.\d+)?)'
}

def process_uploaded_pdf(file):
    """
    Processes an uploaded PDF, handling both text-based and image-based (scanned) files.
    """
    try:
        # Read the file into memory once
        pdf_bytes = file.read()
        
        # --- STRATEGY 1: Try direct text extraction (for text-based PDFs) ---
        try:
            pdf_file_obj = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
            text_from_pypdf2 = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_from_pypdf2 += page_text + "\n"
            
            # If we got a good amount of text, we'll use it
            if len(text_from_pypdf2.strip()) > 50:
                print("PDF appears to be text-based. Using direct extraction.")
                return text_from_pypdf2
            else:
                print("Text extraction yielded little content. May be a scanned PDF.")
        except Exception as e:
            print(f"Direct text extraction failed: {e}")
            pass

        # --- STRATEGY 2: Fallback to OCR (for scanned/image-based PDFs) ---
        if POPPLER_AVAILABLE:
            if not TESSERACT_AVAILABLE:
                return (
                    "OCR requires Tesseract OCR, which is not installed or not on PATH. "
                    "On macOS, install with: brew install tesseract"
                )
            print("Attempting OCR processing with poppler...")
            try:
                images = convert_from_bytes(pdf_bytes)
                ocr_text = ""
                for i, image in enumerate(images):
                    print(f"Processing page {i+1} with OCR...")
                    page_text = pytesseract.image_to_string(image, lang="eng", config="--psm 6")
                    ocr_text += page_text + "\n"
                return ocr_text
            except TesseractNotFoundError:
                return (
                    "Tesseract OCR not found. Install it and ensure it's in PATH. "
                    "On macOS: brew install tesseract"
                )
            except Exception as e:
                print(f"OCR with poppler failed: {e}")
                return "Could not process PDF. It may be scanned or password-protected."
        else:
            print("Poppler not available. Cannot process scanned PDFs.")
            return "Could not process PDF. It may be scanned, and poppler is not installed."

    except Exception as e:
        return f"A critical error occurred while processing the PDF: {e}"

def extract_text_from_image(file):
    """Extracts text from an uploaded image file using Tesseract OCR."""
    if not TESSERACT_AVAILABLE:
        return (
            "Tesseract OCR is not installed or not found on PATH. "
            "On macOS, install with: brew install tesseract"
        )
    try:
        # Normalize to bytes and open via PIL safely
        if hasattr(file, 'read'):
            data = file.read()
        elif isinstance(file, (bytes, bytearray)):
            data = bytes(file)
        else:
            # Assume a filesystem path
            with open(file, 'rb') as f:
                data = f.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        text = pytesseract.image_to_string(image, lang="eng", config="--psm 6")
        return text
    except TesseractNotFoundError:
        return (
            "Tesseract OCR not found. Install it and ensure it's in PATH. "
            "On macOS: brew install tesseract"
        )
    except Exception as e:
        return f"Error processing image: {e}"

def analyze_text(text):
    """Analyzes extracted text for medical values using Regex."""
    print(f"\n=== ANALYZING TEXT ===")
    print(f"Text length: {len(text)} characters")
    print(f"=== PATTERN MATCHING ===")
    
    results = {}
    for key, pattern in patterns.items():
        print(f"Checking {key} with pattern: {pattern}")
        matches = re.findall(pattern, text, re.IGNORECASE)
        print(f"Matches found: {matches}")
        
        if matches:
            try:
                value = float(matches[0])
                normal_range = NORMAL_RANGES[key]['range']
                unit = NORMAL_RANGES[key]['unit']
                if not (normal_range[0] <= value <= normal_range[1]):
                    status = "Abnormal"
                else:
                    status = "Normal"
                results[key] = { 
                    'value': value, 
                    'unit': unit, 
                    'status': status, 
                    'normal_range': f"{normal_range[0]} - {normal_range[1]}" 
                }
                print(f"Added result: {key} = {value}")
            except ValueError as e:
                print(f"Could not convert value to float: {matches[0]}")
    
    print(f"=== ANALYSIS COMPLETE ===")
    return results