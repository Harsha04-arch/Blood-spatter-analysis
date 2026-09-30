import os

frontend_dir = r"c:\Users\heman\OneDrive\Desktop\BPA\frontend"
sidebar_path = os.path.join(frontend_dir, "sidebar.html")

with open(sidebar_path, "r", encoding="utf-8") as f:
    new_sidebar = f.read()

for filename in os.listdir(frontend_dir):
    if not filename.endswith(".html"):
        continue
    if filename in ["index.html", "login.html", "sidebar.html"]:
        continue
        
    filepath = os.path.join(frontend_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Replace Sidebar
    start_idx = content.find('<nav class="sidebar"')
    if start_idx != -1:
        script_start = content.find('<script>', start_idx)
        script_end = content.find('</script>', script_start)
        
        if script_start != -1 and script_end != -1 and 'sidebar-model-badge' in content[script_start:script_end]:
            end_idx = script_end + len('</script>')
        else:
            end_idx = content.find('</nav>', start_idx) + len('</nav>')
            
        content = content[:start_idx] + new_sidebar + content[end_idx:]

    # 2. Text Replacements for Rebranding back to BPA System
    content = content.replace("BLOODTRACE", "BPA System")
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

print("Updated HTML files with BPA System branding and MetaMask sidebar.")
