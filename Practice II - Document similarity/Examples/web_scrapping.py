from bs4 import BeautifulSoup
import requests

# Making a GET request
r = requests.get('https://arxiv.org/list/cs.CL/recent')

soup = BeautifulSoup(r.content, 'html.parser')

html_links = [
	a_tag['href'] 
	for a_tag in soup.find_all('a', {'title': 'View HTML'})
]	

# Print results
if html_links:
	print("Found HTML links:")
	for idx, link in enumerate(html_links, 1):
		print(f"{idx}. {link}")
		r_article = requests.get(link)
		soup_article = BeautifulSoup(r_article.content, 'html.parser')
		# Extract title text
		title = soup_article.title.string.strip() if soup_article.title else None
		print("Extracted Title:", title)

else:
	print("No HTML links found")    
