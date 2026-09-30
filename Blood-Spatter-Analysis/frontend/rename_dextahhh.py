import os

frontend_dir = r"c:\Users\heman\OneDrive\Desktop\BPA\frontend"

for root, dirs, files in os.walk(frontend_dir):
    for filename in files:
        if not (filename.endswith(".html") or filename.endswith(".js") or filename.endswith(".css")):
            continue
            
        filepath = os.path.join(root, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Perform replacements
        original_content = content
        content = content.replace("BPA Forensic System", "dextahhh")
        content = content.replace("BPA System", "dextahhh")
        content = content.replace("BPA_Dataset_2024", "dextahhh_Dataset_2024")
        content = content.replace("BPA", "dextahhh")

        if content != original_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Updated {filename}")

print("Replacement complete.")
