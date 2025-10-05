AI Medical Assistant (Streamlit)

An interactive Streamlit application that combines:

- Symptom Checker using a machine learning model trained on `Training.csv`
- Medical Report Analyzer that extracts text from medical PDFs/images and parses common biomarkers (Hemoglobin, WBC Count, Platelet Count, Glucose)

This tool is intended for educational purposes and is not a substitute for professional medical advice.

---

## Features

- Symptom Checker
  - Multi-select symptoms from `Training.csv`
  - Predict probable condition using a Decision Tree model
  - Pretrained artifacts: `disease_model.pkl`, `label_encoder.pkl`

- Medical Report Analyzer
  - PDF (text-based): direct text extraction via PyPDF2
  - PDF (scanned): OCR via pdf2image + Tesseract OCR (requires Poppler binaries)
  - Images (PNG/JPG): OCR via Tesseract
  - Regex-based parsing for key biomarkers with normal range checks

---

## Project Structure

```
AI_Medical_Project/
├─ app.py                  # Streamlit app
├─ analyzer.py             # PDF/image extraction + OCR + regex analysis
├─ model.py                # Train and persist ML model + label encoder
├─ Training.csv            # Dataset used to train the symptom checker
├─ disease_model.pkl       # Trained Decision Tree model
├─ label_encoder.pkl       # Label encoder for disease names
├─ sample_report.txt       # Sample lab report for testing
├─ test_with_file.py       # Quick test for analysis logic using sample_report.txt
├─ test_patterns.py        # Regex pattern tests
└─ README.md               # This file
```

---

## Requirements

- Python 3.9+ (3.10/3.11 recommended)
- pip
- Tesseract OCR (required for OCR on images and scanned PDFs)
- Poppler binaries (required for converting PDF pages to images for OCR)

Python packages (installed via pip):
- streamlit
- pandas
- scikit-learn
- pillow
- pytesseract
- pdf2image
- PyPDF2

---

## Setup (macOS Intel)

1) Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2) Install Tesseract and Poppler

```bash
brew install tesseract poppler
```

3) Ensure they are on PATH

```bash
which tesseract
which pdftoppm
```

4) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

5) Install Python dependencies

```bash
pip install -U pip
pip install -U streamlit pandas scikit-learn pillow pytesseract pdf2image PyPDF2
```

6) (Optional) Point analyzer to specific Poppler/Tesseract paths

- The app auto-detects common locations. You can override using environment variables:

```bash
# If needed, export path to Poppler binaries (directory containing pdftoppm/pdftocairo)
export POPPLER_PATH="$(dirname "$(which pdftoppm)")"

# Ensure /usr/local/bin is part of PATH (Homebrew on Intel Macs)
export PATH="/usr/local/bin:$PATH"
```

---

## Setup (Linux)

```bash
# Debian/Ubuntu
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -U streamlit pandas scikit-learn pillow pytesseract pdf2image PyPDF2
```

---

## Setup (Windows)

1) Install Tesseract OCR
- Download installer: https://github.com/tesseract-ocr/tesseract
- Add install path to PATH (e.g., `C:\\Program Files\\Tesseract-OCR`)

2) Install Poppler for Windows
- Download binaries: http://blog.alivate.com.au/poppler-windows/
- Add the `bin` directory to PATH (contains `pdftoppm.exe`)

3) Python venv + deps

```powershell
py -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -U streamlit pandas scikit-learn pillow pytesseract pdf2image PyPDF2
```

If Tesseract is not auto-detected by the app, set in code before calling pytesseract:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
```

---

## Running the App

From the project root:

```bash
# Activate your venv first
source .venv/bin/activate  # macOS/Linux
# .\.venv\Scripts\activate  # Windows PowerShell

streamlit run app.py
```

Open the URL displayed in the terminal (typically http://localhost:8501).

---

## Training the Symptom Checker Model

`model.py` trains a Decision Tree on `Training.csv` and saves `disease_model.pkl` and `label_encoder.pkl`.

```bash
# Activate your venv, then
python model.py
```

This will regenerate the model and encoder artifacts in the project directory.

---

## Testing the Analyzer Logic

- Test with a sample text report:

```bash
python test_with_file.py
```

- Test regex patterns quickly:

```bash
python test_patterns.py
```

---

### Your project does use AI/machine learning (plus other practical tools) — let’s break it down by part:

#### Symptom Checker (Explicit AI/Machine Learning Model)
- The symptom checker uses a Decision Tree Classifier (a type of supervised machine learning model, which counts as AI!) that you trained in your model.py file:
You used the disease-symptom dataset (Training.csv) to teach the model to recognize patterns: which combinations of symptoms map to which medical conditions.

- The trained model is saved as disease_model.pkl, and you load it in app.py to make predictions when a user selects symptoms. This is classic predictive AI/ML: the model generalizes from its training data to guess at conditions for new, unseen symptom inputs.

#### Medical Report Analyzer (Uses AI/ML indirectly, plus rule-based tools)
- Tesseract OCR (text extraction from images/PDFs): The tool that reads text from your scanned images/PDFs (Tesseract) uses machine learning (trained neural networks for character recognition, a form of computer vision AI) under the hood to turn visual data (the image of text) into readable text.

- Current value extraction (rule-based, for now): The part that finds values like hemoglobin or PCV uses regex patterns (simple rule-based logic, not custom AI) — but you could upgrade this later to use medical NLP AI models (like BioBERT, a model fine-tuned for medical text) if you wanted to handle more complex/varied report formats reliably.
In short: your symptom checker uses a clear, trained AI/ML model, and the report analyzer relies on an AI-powered OCR tool (plus rule-based extraction right now).

---

## How the Analyzer Works

- For PDFs:
  - Attempts direct text extraction (PyPDF2). If sufficient text (>50 chars) is found, it is used.
  - If not, assumes the PDF is scanned and tries OCR:
    - Converts PDF pages to images (pdf2image + Poppler binaries: `pdftoppm`/`pdftocairo`).
    - Runs Tesseract OCR on each page (`--psm 6`, English).

- For Images (PNG/JPG):
  - Uses Tesseract OCR directly via `pytesseract`.

- Text Analysis:
  - Regex-based extraction for Hemoglobin, WBC Count, Platelet Count, Glucose.
  - Values are checked against simple normal ranges for a Normal/Abnormal status.

Notes:
- The regex approach is heuristic and may misinterpret certain report formats (e.g., reference ranges or scientific notation units). Consider enhancing with line-by-line parsing and unit-aware normalization for production use.

---

## Troubleshooting

- Tesseract not found
  - Ensure it’s installed and on PATH: `which tesseract` (macOS/Linux), or set `pytesseract.pytesseract.tesseract_cmd` on Windows.

- Scanned PDF fails with: "Could not process PDF"
  - Install Poppler and ensure `pdftoppm` is on PATH: `which pdftoppm`.
  - Optionally set `POPPLER_PATH` to the directory containing the Poppler binaries.

- Encrypted/Password-protected PDF
  - The app attempts empty-password decryption. If the PDF is protected, export a decrypted copy and try again.

- Poor OCR quality
  - Ensure the scan is legible and high-resolution.
  - Adjust Tesseract settings (e.g., `--psm 6` works for uniform text; try other PSM modes for different layouts).

- Streamlit cannot find model files
  - Ensure `disease_model.pkl` and `label_encoder.pkl` exist. Re-run `python model.py` to generate them if needed.

---

## Security and Privacy

- Do not upload sensitive PHI to non-secure environments.
- Logs should avoid containing extracted text in production. Consider replacing `print` calls with `logging` and redacting sensitive content.
- This app is for educational purposes and not medical advice.

---

## Configuration

- Environment variables (optional):
  - `POPPLER_PATH`: directory containing Poppler binaries (`pdftoppm`/`pdftocairo`).

- Code-level overrides:
  - `pytesseract.pytesseract.tesseract_cmd` can be set manually to the Tesseract executable if auto-detection fails.

---

## License

No license specified. Add a LICENSE file if you plan to distribute.

---

## Acknowledgements

- Streamlit
- Tesseract OCR / pytesseract
- pdf2image and Poppler
- scikit-learn
- PyPDF2
