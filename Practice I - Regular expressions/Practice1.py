import re
import csv

# Definir etiquetas de equivalencias
tag_equivalences = {}
entry_type_equivalences = {}
with open('tag_equivalence.csv', mode='r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        if row['Type'] == 'Field':
            tag_equivalences[row['BibTeX Field']] = row['RIS Tag']
        elif row['Type'] == 'Entry Type':
            entry_type_equivalences[row['BibTeX Field'].lower()] = row['RIS Tag']

def parse_bibtex(bibtex_str):
    entries = []
    raw_entries = [e for e in re.split(r'(?=@)', bibtex_str) if e.strip()]
    
    for entry in raw_entries:
        match = re.match(r'@(\w+)\s*{\s*([^,]+)\s*,', entry)
        if not match:
            continue
            
        entry_type = match.group(1).lower()
        citation_key = match.group(2).strip()
        
        # Extract fields with improved pattern for multi-line values
        fields_str = entry[match.end():].strip()[:-1]
        fields = {}
        
        # Updated pattern to better handle nested braces and multi-line values
        field_pattern = r'(\w+)\s*=\s*{((?:[^{}]|{(?:[^{}]|{[^{}]*})*})*)}[\s,]*'
        for field_match in re.finditer(field_pattern, fields_str):
            field_name = field_match.group(1).lower()
            field_value = field_match.group(2).strip()
            fields[field_name] = field_value
            
        parsed_entry = {'type': entry_type, 'id': citation_key, 'fields': fields}
        entries.append(parsed_entry)
    
    return entries

def parse_ris(ris_str):
    entries = ris_str.strip().split('ER  - ')
    parsed_entries = []
    for entry in entries:
        if entry.strip():
            lines = entry.strip().split('\n')
            entry_type = ''
            entry_id = ''
            fields = {}
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('TY  - '):
                    entry_type = line[6:]
                elif line.startswith('ID  - '):
                    entry_id = line[6:].strip()
                elif line.startswith('DA  - '):  # Handle date field specifically
                    date_parts = line[6:].strip().split('/')
                    if len(date_parts) >= 1:
                        fields['PY'] = date_parts[0]  # Year
                    if len(date_parts) >= 2:
                        fields['DA_month'] = date_parts[1]  # Month
                    if len(date_parts) >= 3:
                        fields['DA_day'] = date_parts[2]  # Day
                else:
                    tag = line[:2]
                    value = line[6:].strip()
                    if tag in fields:
                        if isinstance(fields[tag], list):
                            fields[tag].append(value)
                        else:
                            fields[tag] = [fields[tag], value]
                    else:
                        if tag == 'AU':
                            fields[tag] = [value]
                        else:
                            fields[tag] = value
            if not entry_id:  # If no ID was found, generate one
                entry_id = "ref_" + str(len(parsed_entries) + 1)
            parsed_entry = {'type': entry_type, 'id': entry_id, 'fields': fields}
            parsed_entries.append(parsed_entry)
    return parsed_entries

def bibtex_to_ris(bibtex_entries):
    month_map = {
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 
        'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
        'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
    }
    
    ris_entries = []
    for entry in bibtex_entries:
        ris_entry = []
        ris_type = entry_type_equivalences.get('@' + entry["type"], "JOUR")
        ris_entry.append(f'TY  - {ris_type}')
        
        # Process authors first
        if 'author' in entry['fields']:
            authors = [author.strip() for author in re.split(r'\s+and\s+|\n\s*and\s*', entry['fields']['author'])]
            for author in authors:
                ris_entry.append(f'AU  - {author}')

        # Process editors
        if 'editor' in entry['fields']:
            editors = [editor.strip() for editor in re.split(r'\s+and\s+|\n\s*and\s*', entry['fields']['editor'])]
            for editor in editors:
                ris_entry.append(f'ED  - {editor}')

        # Add ID
        ris_entry.append(f'ID  - {entry["id"]}')
        
        # Process date fields
        year = entry['fields'].get('year', '')
        month = entry['fields'].get('month', '').lower()
        day = entry['fields'].get('day', '')
        
        if year:
            ris_entry.append(f'PY  - {year}')
            # Construct DA field
            month_num = month_map.get(month[:3], month.zfill(2) if month.isdigit() else '')
            day_num = day.zfill(2) if day else ''
            date_parts = [year]
            if month_num:
                date_parts.append(month_num)
                if day_num:
                    date_parts.append(day_num)
            ris_entry.append(f'DA  - {"/".join(date_parts)}')
        
        # Process remaining fields
        for field, value in entry['fields'].items():
            if field in ['author', 'editor', 'year', 'month', 'day']:  # Skip already processed fields
                continue
            ris_tag = tag_equivalences.get(field)
            if ris_tag:
                if ris_tag == 'SP/EP':
                    pages = value.split('--')
                    ris_entry.append(f'SP  - {pages[0].strip()}')
                    if len(pages) > 1:
                        ris_entry.append(f'EP  - {pages[1].strip()}')
                else:
                    ris_entry.append(f'{ris_tag}  - {value}')
        
        ris_entry.append('ER  - ')
        ris_entries.append('\n'.join(ris_entry))
    
    return '\n'.join(ris_entries)

def ris_to_bibtex(ris_entries):
    bibtex_entries = []
    for entry in ris_entries:
        bibtex_entry = []
        entry_type = next((key[1:] for key, val in entry_type_equivalences.items() if val == entry["type"]), "article")
        bibtex_entry.append(f'@{entry_type}{{{entry["id"]},')
        
        # Handle authors first
        if 'AU' in entry['fields']:
            authors = entry['fields']['AU']
            if isinstance(authors, list):
                author_str = ' and\n'.join(authors)
                bibtex_entry.append(f'author = {{{author_str}}},')
            else:
                bibtex_entry.append(f'author = {{{authors}}},')

        # Handle editors after authors
        if 'ED' in entry['fields']:
            editors = entry['fields']['ED']
            if isinstance(editors, list):
                editor_str = ' and\n'.join(editors)
                bibtex_entry.append(f'editor = {{{editor_str}}},')
            else:
                bibtex_entry.append(f'editor = {{{editors}}},')

        # Process remaining fields
        for tag, value in entry['fields'].items():
            if tag in ['AU', 'ED']:  # Skip already processed fields
                continue
            
            # Handle date components
            if tag == 'DA_month' and value.strip():
                month_map = {'01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', 
                           '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Aug',
                           '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}
                month = month_map.get(value.zfill(2), value)
                bibtex_entry.append(f'month = {{{month}}},')
                continue
                
            if tag == 'DA_day' and value.strip():
                bibtex_entry.append(f'day = {{{value}}},')
                continue
                
            # Handle special cases for address and pages
            if tag == 'CY':  # City/Address field
                bibtex_entry.append(f'address = {{{value}}},')
                continue
                
            if tag in ['SP', 'EP']:  # Page numbers
                if 'SP' in entry['fields'] and 'EP' in entry['fields']:
                    if tag == 'SP':  # Only process once when we see SP
                        bibtex_entry.append(f'pages = {{{entry["fields"]["SP"]}--{entry["fields"]["EP"]}}},')
                continue

            # Handle all other fields
            bibtex_field = next((key for key, val in tag_equivalences.items() if val == tag), None)
            if bibtex_field:
                if isinstance(value, list):
                    combined_value = ' and\n'.join(value)
                    bibtex_entry.append(f'{bibtex_field} = {{{combined_value}}},')
                else:
                    bibtex_entry.append(f'{bibtex_field} = {{{value}}},')

        bibtex_entry[-1] = bibtex_entry[-1].rstrip(',')  # Remove trailing comma from last field
        bibtex_entry.append('}')
        bibtex_entries.append('\n'.join(bibtex_entry))
    return '\n\n'.join(bibtex_entries)

def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def write_file(filename, content):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)

def convert_bibtex_to_ris(input_filename, output_filename):
    bibtex_str = read_file(input_filename)
    bibtex_entries = parse_bibtex(bibtex_str)
    ris_str = bibtex_to_ris(bibtex_entries)
    write_file(output_filename, ris_str)

def convert_ris_to_bibtex(input_filename, output_filename):
    ris_str = read_file(input_filename)
    ris_entries = parse_ris(ris_str)
    bibtex_str = ris_to_bibtex(ris_entries)
    write_file(output_filename, bibtex_str)

#Ejemplo de uso
#convert_bibtex_to_ris('Pruebas1/journal2.bib', 'Pruebas1Test/RIS/journal2.ris')
#convert_ris_to_bibtex('Pruebas1/journal2.ris', 'Pruebas1Test/BibTeX/journal2.bib')

convert_bibtex_to_ris('Pruebas2/BibTeX/journal_test2.bib', 'Pruebas2Test/RIS/journal_test2.ris')
convert_ris_to_bibtex('Pruebas2/RIS/journal_test2.ris', 'Pruebas2Test/BibTeX/journal_test2.bib')