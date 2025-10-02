import re

# Your improved patterns
patterns = {
    'hemoglobin': r'(?:[Hh]emoglobin|[Hh][Gg][Bb])[^\d\n]*([+-]?\d+(?:\.\d+)?)',
    'wbc_count': r'(?:[Ww][Bb][Cc]|[Ww]hite\s*[Bb]lood\s*[Cc]ell)[^\d\n]*([+-]?\d+(?:\.\d+)?)',
    'platelet_count': r'(?:[Pp]latelet|[Pp][Ll][Tt])[^\d\n]*([+-]?\d+(?:\.\d+)?)',
    'glucose': r'(?:[Gg]lucose|[Bb]lood\s*[Ss]ugar)[^\d\n]*([+-]?\d+(?:\.\d+)?)'
}

# Test with various formats
test_cases = [
    "Hemoglobin: 14.2 g/dL",
    "HGB 12.5",
    "WBC Count: 8.5 x10^9/L",
    "White Blood Cell 9.2",
    "Platelet 250",
    "PLT: 220 x10^9/L",
    "Glucose 95 mg/dL",
    "Blood Sugar: 110"
]

print("Testing improved patterns:")
for test_text in test_cases:
    print(f"\nTesting: '{test_text}'")
    for key, pattern in patterns.items():
        match = re.search(pattern, test_text, re.IGNORECASE)
        if match:
            print(f"  {key}: {match.group(1)}")
            