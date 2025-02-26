import re
import csv

# Define tag equivalences
tag_equivalences = {}
with open('tag_equivalence.csv', mode='r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        if row['Type'] == 'Field':
            tag_equivalences[row['BibTeX Field']] = row['RIS Tag']
        elif row['Type'] == 'Entry Type':
            tag_equivalences[row['BibTeX Field']] = row['RIS Tag']

def parse_bibtex(bibtex_str):
    entries = re.split(r'@\w+\{', bibtex_str)[1:]
    parsed_entries = []
    for entry in entries:
        entry_type = re.search(r'^(\w+)', entry).group(1)
        entry_content = re.search(r'\{(.+?),\s*(.+)\}', entry, re.DOTALL)
        entry_id = entry_content.group(1)
        fields_str = entry_content.group(2).strip()
        fields = re.findall(r'(\w+)\s*=\s*\{(.+?)\}', fields_str, re.DOTALL)
        parsed_entry = {'type': entry_type, 'id': entry_id, 'fields': dict(fields)}
        parsed_entries.append(parsed_entry)
    return parsed_entries

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
                if line.startswith('TY  - '):
                    entry_type = line[6:]
                elif line.startswith('ID  - '):
                    entry_id = line[6:]
                else:
                    tag = line[:2]
                    value = line[6:]
                    fields[tag] = fields.get(tag, '') + value
            parsed_entry = {'type': entry_type, 'id': entry_id, 'fields': fields}
            parsed_entries.append(parsed_entry)
    return parsed_entries

def bibtex_to_ris(bibtex_entries):
    ris_entries = []
    for entry in bibtex_entries:
        ris_entry = []
        ris_type = tag_equivalences.get('@' + entry["type"].lower(), "GEN")  # Correctly map entry type
        ris_entry.append(f'TY  - {ris_type}')
        ris_entry.append(f'ID  - {entry["id"]}')
        for field, value in entry['fields'].items():
            ris_tag = tag_equivalences.get(field)
            if ris_tag:
                if ris_tag == 'AU':
                    authors = value.split(' and ')
                    for author in authors:
                        ris_entry.append(f'AU  - {author}')
                elif ris_tag == 'ED':
                    editors = value.split(' and ')
                    for editor in editors:
                        ris_entry.append(f'ED  - {editor}')
                elif ris_tag == 'SP/EP':
                    pages = value.split('--')
                    ris_entry.append(f'SP  - {pages[0]}')
                    if len(pages) > 1:
                        ris_entry.append(f'EP  - {pages[1]}')
                else:
                    ris_entry.append(f'{ris_tag}  - {value}')
        ris_entry.append('ER  - ')
        ris_entries.append('\n'.join(ris_entry))
    return '\n'.join(ris_entries)

def ris_to_bibtex(ris_entries):
    bibtex_entries = []
    for entry in ris_entries:
        bibtex_entry = []
        bibtex_entry.append(f'@{tag_equivalences.get(entry["type"], "misc")}{{{entry["id"]},')
        for tag, value in entry['fields'].items():
            bibtex_field = [key for key, val in tag_equivalences.items() if val == tag]
            if bibtex_field:
                if tag == 'AU':
                    authors = value.replace(' / ', ' and ')
                    bibtex_entry.append(f'  {bibtex_field[0]} = {{{authors}}},')
                elif tag == 'ED':
                    editors = value.replace(' / ', ' and ')
                    bibtex_entry.append(f'  {bibtex_field[0]} = {{{editors}}},')
                elif tag == 'SP' or tag == 'EP':
                    if tag == 'SP':
                        sp = value
                    else:
                        ep = value
                        bibtex_entry.append(f'  pages = {{{sp}--{ep}}},')
                else:
                    bibtex_entry.append(f'  {bibtex_field[0]} = {{{value}}},')
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

# Example usage
convert_bibtex_to_ris('Pruebas1/conference1.bib', 'output.ris')
convert_ris_to_bibtex('input.ris', 'output.bib')