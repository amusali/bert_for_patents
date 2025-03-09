import requests
import time
import random
import re

COMMON_DOMAINS = ['marketscreener.com','markets.ft.com','ft.com','wsj.com','money.cnn.com','zoominfo.com','sec.gov','epa.gov','indeed.com','twitter.com','bloomberg.com','usaspending.gov','finance.yahoo.com','dnb.com','linkedin.com','en.wikipedia.org','wikipedia.com', 'facebook.com', 'wikipedia.org','crunchbase.com', 'pitchbook.com']

# Define the path to API key file and search engine ID
api_key_path = r"C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main\api\google_api_key.txt"
search_engine_id_path = r"C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main\api\search_engine_id.txt"

# Read the API key from the file
with open(api_key_path, 'r') as file:
    api_key = file.read().strip()

# Read the Search Engine ID from the file
with open(search_engine_id_path, 'r') as file:
    search_engine_id = file.read().strip()

def clean_text(s):
    # Remove anything inside parentheses along with the parentheses themselves
    s = re.sub(r'\(.*?\)', '', s)

    # Split the modified string into words
    words = re.findall(r'[\w\.]+|[,\)]', s)

    cleaned_words = []
    for word in words:
        # Keep the word as is if it looks like a URL, otherwise remove unwanted characters
        if re.match(r'\b\w+\.\w+\b', word):
            word = re.sub(r'\.$', '', word)  # Remove trailing period
            cleaned_words.append(word)
        else:
            word = word.replace(",", "").replace(".", "")
            cleaned_words.append(word)
    
    return ' '.join(' '.join(cleaned_words).split())






saved_api = 0

def remove_common_endings(company_name):
    company_name = clean_text(company_name)
    endings = ['inc', 'inc.', 'corp', 'corp.', 'comp',
               'ltd', 'na', 'lp', 'co', 'corporation', 
               'llc', 'partners', ',inc', '.inc', 'inc-',
                'fund', 'acq', 'acquisition', 'acquisitions', 'asst', 'asset', 'assets', '-asst']  # Add more common endings as needed
    unique_names = []
    lowercase_name = company_name.lower()
    
    
    stripped_name = lowercase_name.strip()
    
    while stripped_name.split()[-1] in endings:
        stripped_name = ' '.join(stripped_name.split()[:-1]).strip()
    unique_names.append(stripped_name)
    #unique_names.append(lowercase_name.strip())
    return unique_names

def perform_google_search(company_name, matched_names, api_key = api_key, search_engine_id = search_engine_id):
    #true_names = []
    start_time = time.time()
    api_count = 0
   # for x in matched_names:
    #    true_names.append(str(x[0]))
    #matched_names = true_names
    #print(matched_names)

    # Perform the search for the company name
    company_search_url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={company_name}"
    company_search_response = requests.get(company_search_url)
    api_count += 1
    company_search_results = company_search_response.json().get("items", [])

    # Perform the search for each matched name and store the results
    matched_results = []
    successful_matches = []
    for matched_name in matched_names:
        if remove_common_endings(clean_text(matched_name)) == remove_common_endings(clean_text(company_name)):
            successful_matches.append(matched_name)
            print('A Saved API')
            continue
        matched_search_url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={matched_name}"
        matched_search_response = requests.get(matched_search_url)
        #print(matched_search_response)
        api_count += 1

        matched_search_results = matched_search_response.json().get("items", [])
        #print(matched_search_results)
        #print(matched_search_results)
        matched_results.append({
            "matched_name": matched_name,
            "search_results": matched_search_results
        })

    # Determine successful matches based on common search results and common domains
    #print(matched_results)
    for result in matched_results:
        common_results = set()
        for company_result in company_search_results:
            for matched_result in result["search_results"]:
                if company_result["link"] == matched_result["link"]:
                    common_results.add(result["matched_name"])

        if len(common_results) >= 1:
            successful_matches.append(result["matched_name"])
   # print(common_results)
    # Check if there are any leftover alleged matches
    leftover_matches = set(matched_names) - set(successful_matches)
    for result in matched_results:
        if result["matched_name"] in leftover_matches:
            common_domains = get_common_domains(company_search_results, result["search_results"])
            if common_domains:
                successful_matches.append(result["matched_name"])

    end_time = time.time()
    #print(f"It took {end_time - start_time:.2f} secs to find the successfull match.")


    return successful_matches, api_count


def get_common_domains(company_results, matched_results):
    common_domains = set()
    for company_result in company_results:
        for matched_result in matched_results:
            if is_common_domain(company_result["link"], matched_result["link"]):
                common_domains.add(matched_result["link"])

    return common_domains


def is_common_domain(company_link, matched_link):
    company_domain = get_domain(company_link)
    matched_domain = get_domain(matched_link)
    
    if company_domain in COMMON_DOMAINS or matched_domain in COMMON_DOMAINS:
        return False
    #if company_domain == matched_domain:
        print("THE SAME DOMAIN: ", matched_domain)
    return company_domain == matched_domain


def get_domain(url):
    start_pos = url.find('//') + 2
    end_pos = url.find('/', start_pos)
    domain = url[start_pos:end_pos]
    
    if domain.startswith("www."):
        domain = domain[4:]  # Remove "www" from the beginning
    
    return domain
