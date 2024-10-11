import requests
from bs4 import BeautifulSoup

class CustomError(Exception):
    pass

def fetch_description(patent_id):
    
    # Modify the ID to be in  Google Patent style
    url = f'https://patents.google.com/patent/US{patent_id}/en'

    # Send a GET request to the webpage
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:

        # Parse the HTML content using Beautiful Soup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        descriptions = [description.get_text(strip=True) for description in soup.select("div.description, div.description-line, div.description-paragraph")]

        if descriptions:
            return descriptions
        else:
            raise CustomError("No Description found")
    else:
        print('Failed to retrieve the webpage')


def fetch_claims(patent_id):
    
    # Modify the ID to be in  Google Patent style
    url = f'https://patents.google.com/patent/US{patent_id}/en'

    # Send a GET request to the webpage
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:

        # Parse the HTML content using Beautiful Soup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        claims = [claim.get_text(strip=True) for claim in soup.select("div.claims, ol.claims")]

        if claims:
            return claims
        else:
            raise CustomError("No Claims found")
    else:
        print('Failed to retrieve the webpage')
