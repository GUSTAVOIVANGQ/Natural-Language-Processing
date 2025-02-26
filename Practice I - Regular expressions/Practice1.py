import re
import csv
import os

class BibTexRisConverter:
    def __init__(self, tag_equivalence_file):
        # Load tag equivalences from CSV
        self.bibtex_to_ris = {}  # Maps BibTeX fields to RIS tags
        self.ris_to_bibtex = {}  # Maps RIS tags to BibTeX fields
        self.entry_types = {}    # Maps BibTeX entry types to RIS types
        self.ris_entry_types = {} # Maps RIS types to BibTeX entry types
        
        with open(tag_equivalence_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Type'] == 'Field':
                    bibtex_field = row['BibTeX Field'].strip()
                    ris_tag = row['RIS Tag'].strip()
                    self.bibtex_to_ris[bibtex_field] = ris_tag
                    self.ris_to_bibtex[ris_tag] = bibtex_field
                elif row['Type'] == 'Entry Type':
                    bibtex_type = row['BibTeX Field'].strip()
                    ris_type = row['RIS Tag'].strip()
                    self.entry_types[bibtex_type.replace('@', '')] = ris_type
                    self.ris_entry_types[ris_type] = bibtex_type.replace('@', '')
    
    def bibtex_to_ris_convert(self, bibtex_content):
        """Convert BibTeX content to RIS format"""
        # Extract each BibTeX entry
        entries = re.findall(r'@(\w+)\{(.*?),\s*(.*?)\s*\}', bibtex_content, re.DOTALL)
        
        ris_output = []
        
        for entry_type, citation_key, fields_text in entries:
            # Start with the entry type
            ris_entry = []
            if entry_type.lower() in self.entry_types:
                ris_entry.append(f"TY  - {self.entry_types[entry_type.lower()]}")
            else:
                ris_entry.append(f"TY  - JOUR")  # Default to journal if type not found
            
            # Add the citation key as ID
            ris_entry.append(f"ID  - {citation_key}")
            
            # Extract and convert each field
            fields = re.findall(r'(\w+)\s*=\s*\{(.*?)\}(?:,|\s*$)', fields_text, re.DOTALL)
            
            for field_name, field_value in fields:
                field_name = field_name.lower()
                
                if field_name == 'pages':
                    # Handle pages special case (split into SP and EP)
                    pages = re.split(r'[-–—]', field_value)
                    if len(pages) >= 1:
                        ris_entry.append(f"SP  - {pages[0].strip()}")
                    if len(pages) >= 2:
                        ris_entry.append(f"EP  - {pages[1].strip()}")
                elif field_name == 'author' or field_name == 'editor':
                    # Split authors by 'and'
                    authors = [author.strip() for author in re.split(r'\s+and\s+', field_value)]
                    ris_tag = self.bibtex_to_ris.get(field_name, field_name.upper())
                    for author in authors:
                        ris_entry.append(f"{ris_tag}  - {author}")
                elif field_name == 'keywords':
                    # Split keywords
                    keywords = [kw.strip() for kw in re.split(r',', field_value)]
                    for keyword in keywords:
                        ris_entry.append(f"KW  - {keyword}")
                else:
                    # General case
                    ris_tag = self.bibtex_to_ris.get(field_name, field_name.upper())
                    ris_entry.append(f"{ris_tag}  - {field_value}")
            
            # End of reference
            ris_entry.append("ER  -")
            ris_entry.append("")  # Empty line between references
            
            ris_output.append("\n".join(ris_entry))
        
        return "\n".join(ris_output)
    
    def ris_to_bibtex_convert(self, ris_content):
        """Convert RIS content to BibTeX format"""
        # Split content into separate references (blocks between TY and ER)
        references = re.split(r'ER  -\s*', ris_content)
        
        bibtex_output = []
        
        for ref in references:
            if not ref.strip():
                continue
                
            # Extract fields
            fields = re.findall(r'(\w+)\s*-\s*(.*?)(?:\r?\n|$)', ref)
            
            entry_type = "article"  # Default
            citation_key = ""
            bibtex_fields = []
            
            authors = []
            start_page = ""
            end_page = ""
            
            for tag, value in fields:
                tag = tag.strip()
                value = value.strip()
                
                if tag == "TY":
                    # Set entry type
                    if value in self.ris_entry_types:
                        entry_type = self.ris_entry_types[value]
                    else:
                        entry_type = "article"  # Default
                elif tag == "ID":
                    citation_key = value
                elif tag == "AU":
                    # Collect authors
                    authors.append(value)
                elif tag == "SP":
                    start_page = value
                elif tag == "EP":
                    end_page = value
                else:
                    # Standard field conversion
                    if tag in self.ris_to_bibtex:
                        bibtex_field = self.ris_to_bibtex[tag]
                        bibtex_fields.append(f"  {bibtex_field} = {{{value}}}")
            
            # Process authors
            if authors:
                bibtex_fields.insert(0, f"  author = {{{' and '.join(authors)}}}")
            
            # Process pages
            if start_page or end_page:
                if start_page and end_page:
                    bibtex_fields.append(f"  pages = {{{start_page}--{end_page}}}")
                elif start_page:
                    bibtex_fields.append(f"  pages = {{{start_page}}}")
            
            # Create BibTeX entry
            bibtex_entry = [f"@{entry_type}{{{citation_key},"]
            bibtex_entry.extend(bibtex_fields)
            bibtex_entry.append("}")
            
            bibtex_output.append("\n".join(bibtex_entry))
        
        return "\n\n".join(bibtex_output)
    
    def convert_file(self, input_file, output_file=None):
        """Convert a file from BibTeX to RIS or vice versa"""
        file_extension = os.path.splitext(input_file)[1].lower()
        
        if not output_file:
            # Generate output filename
            if file_extension == '.bib':
                output_file = input_file.replace('.bib', '.ris')
            elif file_extension == '.ris':
                output_file = input_file.replace('.ris', '.bib')
            else:
                output_file = input_file + '.converted'
        
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Determine conversion direction
        if file_extension == '.bib':
            result = self.bibtex_to_ris_convert(content)
        elif file_extension == '.ris':
            result = self.ris_to_bibtex_convert(content)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        
        return output_file


def main():
    # Get tag equivalence file path
    tag_file = "tag_equivalence.csv"
    
    converter = BibTexRisConverter(tag_file)
    
    # Ask for input file
    input_file = input("Enter the input file path: ")
    
    try:
        output_file = converter.convert_file(input_file)
        print(f"Conversion successful! Output file: {output_file}")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")


if __name__ == "__main__":
    main()