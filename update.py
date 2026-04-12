import urllib.request
import os
import re

def update_publications():
    # Fallback to broader query if strict au: fails
    query = 'Azime'
    api_url = f"https://export.arxiv.org/api/query?search_query=all:{query}&sortBy=submittedDate&sortOrder=descending&max_results=100"
    
    # confirmed author substrings
    confirmed_names = ["azime"]
    
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
            is_match = False
            for auth in authors_xml:
                name_m = re.search(r'<name>(.*?)</name>', auth)
                if name_m:
                    name = name_m.group(1).strip()
                    authors_list.append(name)
                    if any(c_name.lower() in name.lower() for c_name in confirmed_names):
                        is_match = True
            
            if not is_match:
                skipped += 1
                continue

            title_m = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            date_m = re.search(r'<published>(.*?)</published>', entry)
            sum_m = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            id_m = re.search(r'<id>(.*?)</id>', entry)

            if title_m and date_m and sum_m and id_m:
                title = title_m.group(1).strip().replace('\n', ' ').replace('"', '\\"')
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
        
        print(f"Successfully cached {count} publications for 'Azime' in {output_dir}/")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    update_publications()
