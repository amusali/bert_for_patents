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
import random
from numpy.linalg import norm
import numpy as np 
import tensorflow as tf
import json
from path_utils import get_base_path



## Get base path depending on the environment
base_path = get_base_path()


if 'COLAB_GPU' in os.environ:
    api_key = os.getenv('patentsview_api_key')
else:
    # Construct file paths
    api_key_path = os.path.join(base_path, '05 Analysis/01 Main/api/api_key.txt')

    # Read the API key from the file
    with open(api_key_path, 'r') as file:
        api_key = file.read().strip()


# Build URLs
headers = {
    "X-Api-Key": api_key
}

ua = UserAgent()

## Base URL
def get_patents(company_name, date, source = 'legacy'):
    # Replace '&' symbol in company_name for URL encoding
    if "&" in company_name:
        company_name = company_name.replace("&", "%26")
        print(company_name)
    
    closest_match = company_name
    first_year = 1990  # Define the earliest year for filtering patents

    # Source: Assignee, querying PatentsView API for patents assigned to a company
    if source == 'assignee':
        patent_base_url = 'https://search.patentsview.org/api/v1/patent/?q='
        assignee_field = 'assignees.assignee_organization'
        date_field = 'application.filing_date'
        fields = '["cpc_current.cpc_group_id","patent_num_times_cited_by_us_patents","application.filing_date","patent_id","assignees.assignee_country", "patent_abstract"]'
        
        # Build the query URL with company name and fields
        patent_url = f'{patent_base_url}{{"{assignee_field}": "{closest_match}"}}&f={fields}'
        print(patent_url)

        # Initialize empty lists to store patents before and after the given date
        patents_before = []
        patents_after = []
        
        # Convert the input date to datetime object for comparison
        date = datetime.strptime(date, "%Y-%m-%d")

        while True:
            # Adding random sleep to avoid rate-limiting (sleep for a random time between requests)
            time.sleep(random.random())

            # Make the request to the PatentsView API
            patent_response = requests.get(patent_url, headers=headers)
            print(patent_url)
            
            # If the request is successful
            if patent_response.status_code == 200:
                patent_data = patent_response.json()
                patents = patent_data['patents']  # Extract patents from response
                
                # If patents are returned in the response
                if patents:
                    for patent in patents:
                        # Skip patents without CPC classification or application date
                        if 'cpc_current' not in patent:
                            continue
                        if 'application' not in patent:
                            continue
                        
                        # Process the patent's application filing date
                        if patent['application'][0]['filing_date'] is not None:
                            patent_date = datetime.strptime(patent['application'][0]['filing_date'], "%Y-%m-%d")
                            
                            # Skip patents before the first_year threshold
                            if int(patent_date.year) < first_year:
                                continue
                            
                            # Extract relevant fields: citations, tech field, year, and abstract
                            citations = patent['patent_num_times_cited_by_us_patents']
                            tech_field = patent['cpc_current'][0]['cpc_group_id']
                            patent_year = patent_date.year
                            to_add = {'patent_id':patent['patent_id'], 'citations': citations, 'tech_field': tech_field, 'year': patent_date, "abstract" : patent['patent_abstract']}
                           
                            # Categorize patents based on the comparison to the provided date
                            if patent_date < date:
                                patents_before.append(to_add)
                            else:
                                patents_after.append(to_add)

                        # Handle patents with missing filing dates
                        elif patent['application'][0]['filing_date'] is None:
                            print("!!!! NONE DATE ATTENTION !!!!")
                        
                        # Ensure no duplicate processing based on patent ID
                        elif patent['patent_id'] < last_patent_id:
                            citations = patent['patent_num_times_cited_by_us_patents']
                            tech_field = patent['cpc_current'][0]['cpc_group_id']
                            patent_year = patent_date.year
                            to_add = {'patent_id':patent['patent_id'], 'citations': citations, 'tech_field': tech_field, 'year': patent_date}             
                            patents_before.append(to_add)
                    
                    # Update the last patent ID and modify the URL for the next batch of patents
                    last_patent_id = patents[-1]['patent_id']
                    patent_url = f'{patent_base_url}{{"{assignee_field}": "{closest_match}"}}&f={fields}&o={{"size":1000, "after":"{last_patent_id}"}}&s=[{{"patent_id":"asc"}}]'
                    print("Size of patents before: ", len(patents_before), "Size of patents after: ", len(patents_after))

                else:
                    # Break the loop if no more patents are returned
                    break
            else:
                # Handle API errors and print the error code
                print("Error: Could not retrieve patent data from ASSIGNEES. Error code is:", patent_response.status_code)
                return None

        # Return patents before and after the provided date
        return patents_before, patents_after

    # Source: Legacy, querying PatentsView API for patents assigned to a company (legacy data format)
    elif source == 'legacy':
        patent_base_url = 'https://api.patentsview.org/patents/query?o={"page":1,"per_page":10000}&q='
        assignee_field = 'assignee_organization'
        date_field = 'app_date'
        fields = '["cpc_subgroup_id", "cpc_group_id" ,"app_date","patent_id", "patent_date","cpc_category","assignee_country", "patent_num_cited_by_us_patents", "patent_abstract", "citedby_patent_id", "citedby_patent_title", "citedby_patent_date"]'

        # Build the query URL with company name and fields
        patent_url = f'{patent_base_url}{{"{assignee_field}": "{closest_match}"}}&f={fields}'
        print(patent_url)

        # Initialize empty lists to store patents before and after the given date
        patents_before = []
        patents_after = []
        
        # Convert the input date to datetime object for comparison
        date = datetime.strptime(date, "%Y-%m-%d")
        last_page = 1  # Keep track of the pagination

        while True:
            # Simulate real browser requests using random headers to avoid blocks
            fake_headers = {'User-Agent': ua.random}

            # Make the request to the PatentsView API
            patent_response = requests.get(patent_url, headers=fake_headers)
            
            # If the request is successful
            if patent_response.status_code == 200:
                patent_data = patent_response.json()
                patents = patent_data['patents']  # Extract patents from response
                
                # If patents are returned in the response
                if patents:
                    for patent in patents:
                        # Skip patents without application date
                        if patent['applications'][0][date_field] is not None:
                            patent_date = datetime.strptime(patent['applications'][0][date_field], "%Y-%m-%d")
                            
                            # Skip patents before the first_year threshold
                            if int(patent_date.year) < first_year:
                                continue
                            
                            # Extract relevant fields: citations, tech field, year, and abstract
                            citations = patent['patent_num_cited_by_us_patents']
                            tech_field_subgroup = patent['cpcs'][0]['cpc_subgroup_id']
                            tech_field_group = patent['cpcs'][0]['cpc_group_id']

                            abstract = patent['patent_abstract']
                            citedby_patents = patent['citedby_patents']
                            patent_year = patent_date.year
                            to_add = {'patent_id':patent['patent_id'], 'citations': citations, 'tech_field_subgroup': tech_field_subgroup, 'tech_field_group': tech_field_group, 'year': patent_date, 'abstract': abstract, 'citedby_patents' : citedby_patents}  
                           
                            # Categorize patents based on the comparison to the provided date
                            if patent_date < date:
                                patents_before.append(to_add)
                            else:
                                patents_after.append(to_add)
                        else:
                            continue
                    
                    # Update the last page and modify the URL for the next page of results
                    last_page += 1
                    patent_url = f'https://api.patentsview.org/patents/query?o={{"page":{last_page},"per_page":10000}}&q={{"{assignee_field}": "{closest_match}"}}&f={fields}'
                    print("Size of patents before: ", len(patents_before), "Size of patents after: ", len(patents_after))

                # Break the loop if fewer than 10,000 patents are returned, indicating the last page
                if patent_data['count'] < 10000:
                    break
                
            else:
                # Handle API errors and print the error code
                print("Error: Could not retrieve patent data from LEGACY. Code is: ", patent_response.status_code)
                print('Problematic URL: ', patent_url)
                return None

        # Return patents before and after the provided date
        return patents_before, patents_after

    
def get_patents_from_fields(field, year, group_only = False):
    year = str(year)
    
    analyzed_fields = {}

    if field not in analyzed_fields:
        analyzed_fields[field] = {}

    if year not in analyzed_fields[field]:
        analyzed_fields[field][year] = {}

        # Construct the query URL
        base_url = 'https://api.patentsview.org/patents/query?o={"page":1,"per_page":10000}&q='
        
        ## Change URL based on the field requested
        if group_only:

            query = f'{{"_and":[{{"_and":[{{"cpc_group_id":"{field}"}},{{"cpc_sequence":0}}]}},{{"_gte":{{"app_date":"{year}-01-01"}}}},{{"_lte":{{"app_date":"{year}-12-31"}}}}]}}'

        else:
            query = f'{{"_and":[{{"_and":[{{"cpc_subgroup_id":"{field}"}},{{"cpc_sequence":0}}]}},{{"_gte":{{"app_date":"{year}-01-01"}}}},{{"_lte":{{"app_date":"{year}-12-31"}}}}]}}'

        fields = '["patent_num_cited_by_us_patents", "patent_id", "patent_abstract", "assignee_organization", "citedby_patent_id", "citedby_patent_title", "citedby_patent_date"]'
        full_url = f'{base_url}{query}&f={fields}'

        last_page = 1
        analyzed_fields[field][year]["patents"] = []

        while True:
            headers = {'User-Agent': ua.random}  # Randomize user-agent to avoid blocking
            tt = random.random()  # Random sleep time between requests
            print('Sleeping for: ', tt, 'seconds')
            time.sleep(tt + 2)
            
            response = requests.get(full_url, headers=headers)
            if response.status_code == 200:
                last_page += 1
                data = response.json()
                patents = data['patents']
                if not patents:
                    print('NO PATENTS HERE')
                    analyzed_fields[field][year] = None
                    return None
                for patent in patents:
                    # Skip patents without abstracts
                    if patent['patent_abstract'] is None:
                        continue
                    # Append relevant patent information
                    analyzed_fields[field][year]["patents"].append(
                        {"patent_id": patent['patent_id'],
                         "citations": patent['patent_num_cited_by_us_patents'],
                         "abstract": patent['patent_abstract'],
                         "assignee_organization": patent['assignees'][0]['assignee_organization'],
                         "citedby_patents" : patent['citedby_patents']}
                    )
                # Update the URL for the next page
                half_url = f'https://api.patentsview.org/patents/query?o={{"page":{last_page},"per_page":10000}}&q='
                full_url = f'{half_url}{query}&f={fields}'
                
                if data['count'] < 10000:  # Break the loop if fewer than 10,000 patents are returned
                    break
            else:
                print('Error code is: ', response.status_code)
                analyzed_fields[field][year] = None
                return None

        count = len(analyzed_fields[field][year]['patents'])  # Total number of patents
        return analyzed_fields, count
