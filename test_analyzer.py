from analyzer import process_uploaded_pdf, extract_text_from_image, analyze_text

# Test with a PDF file
with open('cbc.pdf', 'rb') as f:
    pdf_text = process_uploaded_pdf(f)
    print("PDF Text:")
    print(pdf_text)

    results = analyze_text(pdf_text)
    print("\nAnalysis Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

# Test with an image file
image_text = extract_text_from_image('cbc.png')
print("\nImage Text:")
print(image_text)

results = analyze_text(image_text)
print("\nAnalysis Results:")
for key, value in results.items():
    print(f"{key}: {value}")
