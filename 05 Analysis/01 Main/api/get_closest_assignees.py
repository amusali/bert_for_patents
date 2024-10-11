# Import modules
import os
import requests
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from datetime import datetime, timedelta
from urllib.parse import parse_qs, unquote, urlparse
import re
import time
from fake_useragent import UserAgent
import random
import time
import pandas as pd
import json
from api.utils import remove_common_endings



def get_closest_assignees(company_name, threshold=85):
    # Define the path to API key file
    api_key_path = r"C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main\api\api_key.txt"

    # Read the API key from the file
    with open(api_key_path, 'r') as file:
        api_key = file.read().strip()

    # Build URLs
    headers = {
        "X-Api-Key": api_key
    }

    ua = UserAgent()

    # URLs
    assignee_base_url = 'https://search.patentsview.org/api/v1/assignee/?q='
    legacy_base_url = 'https://api.patentsview.org/assignees/query?q='
    assignee_urls = []
    company_names = remove_common_endings(company_name)
    print("Company names are: ", company_names)

    assignee_orgs = set()
    best_matches = []
    for name in company_names:
        assignee_url_begins = f'{assignee_base_url}{{"_begins":{{"assignee_organization":"{name}"}}}}&f=["assignee_id","assignee_lastknown_country","assignee_organization"]'
        legacy_url_begins = f'{legacy_base_url}{{"_begins":{{"assignee_organization":"{name}"}}}}&f=["assignee_id","assignee_lastknown_country","assignee_organization"]'
        
        assignee_urls.append(legacy_url_begins)
        
        
        assignee_url_contains = f'{assignee_base_url}{{"_contains":{{"assignee_organization":"{name}"}}}}&f=["assignee_id","assignee_lastknown_country","assignee_organization"]'
        legacy_url_contains = f'{legacy_base_url}{{"_contains":{{"assignee_organization":"{name}"}}}}&f=["assignee_id","assignee_lastknown_country","assignee_organization"]'
        assignee_urls.append(legacy_url_contains)
        assignee_urls.append(assignee_url_begins)
        assignee_urls.append(assignee_url_contains)
        

    for url in assignee_urls:
        fake_headers = {'User-Agent': ua.random}
        assignee_org_strings = [org_tuple[0].lower() for org_tuple in assignee_orgs]
        new_ones = []
        source = 'legacy' if url.startswith(legacy_base_url) else 'assignee'
       # print('Currently getting the URL from source: ',source,  url)
        time.sleep(random.random())
        if source == 'legacy':
            assignee_response = requests.get(url, headers = fake_headers)
            query_params = parse_qs(urlparse(url).query)
            company_name_being_used = query_params['q'][0].split('assignee_organization":"')[1].split('"}')[0]
        else:
            assignee_response = requests.get(url, headers=headers)
            query_params = parse_qs(urlparse(url).query)
            company_name_being_used= unquote(query_params['q'][0]).split('assignee_organization":"')[1].split('"}}')[0]
        
        #print("The code is: ", assignee_response.status_code)
        if assignee_response.status_code == 200:
            assignee_data = assignee_response.json()
            #print(assignee_data)
            assignees = assignee_data['assignees']
            if assignees is None:
                #print('NONE ASSIGNEES')
                continue
            for assignee in assignees:
                if assignee['assignee_lastknown_country'] in ['US', 'America'] and assignee['assignee_organization'].lower() not in assignee_org_strings:
                    # Each element is a tuple: (assignee_organization, source)
                    new_ones.append(assignee['assignee_organization'])
                    assignee_orgs.add((assignee['assignee_organization'], source))
                    #print('Adding', (assignee['assignee_organization'], source), ' from the source: ', source)
            if new_ones:
                if len(new_ones) == 1:
                    match = process.extract(company_name_being_used, [new_ones])
                    if match[0][1] > threshold and re.search(r'\b' + company_name_being_used + r'\b', match[0][0][0], re.IGNORECASE) is not None:
                       # print(match)
                        #print("A UNIQUE COMPANY")
                        tpl = (match[0][0][0], source)
                        best_matches.append(tpl) 
                        #print(name, ' is being matched to', match)

                else:
                    matches = process.extract(company_name_being_used, new_ones)
                    matches_above_threshold = [(match[0], source) for match in matches if match[1] >= threshold
                                               and re.search(r'\b' + company_name_being_used + r'\b', match[0], re.IGNORECASE) is not None]  
                    #exact_matches = [match for match in matches_above_threshold if company_name_being_used.lower() in match[0].lower()]

                    best_matches.extend(matches_above_threshold)    
                    #print(name, ' is being matched to', matches_above_threshold)
    if assignee_orgs:
        
        print('All THE FOUND COMPANIES ARE: ', assignee_orgs)
        if best_matches:
            print("Best matches are: ", best_matches)

            return best_matches
        else:
            print("No best matches found.")
            return None
    else:
        print('NO Assignees Found')
        return None
        