from analyzer import analyze_text

# Read and analyze the sample report
with open('sample_report.txt', 'r') as f:
    text = f.read()

print("Analyzing sample report...")
results = analyze_text(text)

print("\n=== RESULTS ===")
for key, result in results.items():
    print(f"{key.replace('_', ' ').title()}: {result['value']} {result['unit']} (Status: {result['status']})")
    print(f"  Normal Range: {result['normal_range']}")