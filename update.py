import urllib.request
import os
import re

def update_publications():
    # Fallback to broader query if strict au: fails
    query = 'Azime'
    api_url = f"https://export.arxiv.org/api/query?search_query=all:{query}&sortBy=submittedDate&sortOrder=descending&max_results=100"
    
    # Strictly your name variations
    my_names = ["Israel Abebe Azime", "Israel A. Azime", "I. A. Azime", "Israel Abebe"]
    
    # Exclude other authors with similar name substrings
    exclude_authors = ["Azime Tarhan", "Tarhan, A."]
    
    # Exclude specific keywords that match false positive papers
    exclude_keywords = ["extending modules", "torsion theory", "cyclic τ-nonsingular"]
    
    print(f"Fetching from {api_url}...")
    
    try:
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        data = response.read().decode('utf-8')
        
        entries = re.findall(r'<[a-z:]*entry>(.*?)</[a-z:]*entry>', data, re.DOTALL | re.IGNORECASE)
        
        if not entries:
            print("No entries found at all in response.")
            return

        output_dir = '_publications'
        os.makedirs(output_dir, exist_ok=True)
        
        # Clear existing
        for f in os.listdir(output_dir):
            if f.endswith('.md'):
                os.remove(os.path.join(output_dir, f))

        count = 0
        skipped = 0
        for entry in entries:
            authors_xml = re.findall(r'<author>(.*?)</author>', entry, re.DOTALL)
            authors_list = []
            is_me = False
            is_other_azime = False
            
            for auth in authors_xml:
                name_m = re.search(r'<name>(.*?)</name>', auth)
                if name_m:
                    name = name_m.group(1).strip()
                    authors_list.append(name)
                    # Check if it's strictly you
                    if any(my_name.lower() in name.lower() for my_name in my_names):
                        is_me = True
                    # Check if it's the other known Azime
                    if any(other.lower() in name.lower() for other in exclude_authors):
                        is_other_azime = True
            
            title_m = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            title_text = title_m.group(1).strip() if title_m else ""
            
            # Final check logic: Must be me, must NOT be other Azime, and must NOT match false positive keywords
            if is_me and not is_other_azime:
                if any(key.lower() in title_text.lower() for key in exclude_keywords):
                    skipped += 1
                    continue
            else:
                skipped += 1
                continue

            date_m = re.search(r'<published>(.*?)</published>', entry)
            sum_m = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            id_m = re.search(r'<id>(.*?)</id>', entry)

            if title_m and date_m and sum_m and id_m:
                title = title_text.replace('\n', ' ').replace('"', '\\"')
                date = date_m.group(1)[:10]
                summary = sum_m.group(1).strip().replace('\n', ' ').replace('"', '\\"')
                paper_url = id_m.group(1).strip()
                authors_str = ", ".join(authors_list).replace('"', '\\"')
                
                clean_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_').lower()[:60]
                filename = f"{date}_{clean_title}.md"
                
                with open(os.path.join(output_dir, filename), 'w') as f:
                    f.write('---\n')
                    f.write(f'title: "{title}"\n')
                    f.write(f'date: {date}\n')
                    f.write(f'paper: {paper_url}\n')
                    f.write(f'authors: "{authors_str}"\n')
                    f.write('layout: default\n')
                    f.write('---\n\n')
                    f.write(summary)
                count += 1
        
        print(f"Successfully cached {count} confirmed publications for Israel Azime.")
        print(f"Filtered out {skipped} irrelevant papers.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    update_publications()
