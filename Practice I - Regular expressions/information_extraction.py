import re

info = "author = {Torres, Miguel and Cecilia, Bueno}"

pattern = r"\{(.*?)\}"

match = re.search(pattern, info)

if match:
    extracted_info = match.group(1)
    print(extracted_info)
else:
    print("No match found.")

names = re.split("and", extracted_info)
print (names)
