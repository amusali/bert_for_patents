# Import modules
import os
import requests
from datetime import datetime
import time
from fake_useragent import UserAgent
import random
import time
import random
from path_utils import get_base_path
import api.patent as apipat
import importlib
importlib.reload(apipat)
from api.patent import Patent, CitedByPatent


## Get base path depending on the environment
base_path = get_base_path()


## Read API key for patentsview
if 'COLAB_GPU' in os.environ:
    api_key = os.getenv('patentsview_api_key')
else:
    # Construct file paths
    api_key_path = os.path.join(base_path, '05 Analysis/01 Main/api/api_key.txt')

    # Read the API key from the file
    with open(api_key_path, 'r') as file:
        api_key = file.read().strip()

    scraperapi_key_path = os.path.join(base_path, "05 Analysis/01 Main/api/scraperapikey.txt")
    # Read the API key from the file
    with open(scraperapi_key_path, 'r') as file:
        scraperapi_key = file.read().strip()

# Build Headers
headers = {
    "X-Api-Key": api_key
}

ua = UserAgent()


def get_patents(company_name, date, source='legacy'):
    if "&" in company_name:
        company_name = company_name.replace("&", "%26")
        print(company_name)
    
    closest_match = company_name
    first_year = 1990  # Filter for patents from 1990 onwards

    # Source: 'assignee'
    if source == 'assignee':
        patent_base_url = 'https://search.patentsview.org/api/v1/patent/?q='
        assignee_field = 'assignees.assignee_organization'
        fields = '["cpc_current.cpc_group_id","patent_num_times_cited_by_us_patents","application.filing_date","patent_id","assignees.assignee_country", "patent_abstract"]'
        patent_url = f'{patent_base_url}{{"{assignee_field}": "{closest_match}"}}&f={fields}'
        print(patent_url)

        patents_before = []
        patents_after = []
        date = datetime.strptime(date, "%Y-%m-%d")

        while True:
            time.sleep(random.random())
            patent_response = requests.get(patent_url)
            print(patent_url)
            
            if patent_response.status_code == 200:
                patent_data = patent_response.json()
                patents = patent_data['patents']
                
                if patents:
                    for patent in patents:
                        if 'cpc_current' not in patent or 'application' not in patent:
                            continue
                        
                        if patent['application'][0]['filing_date']:
                            patent_date = datetime.strptime(patent['application'][0]['filing_date'], "%Y-%m-%d")
                            if patent_date.year < first_year:
                                continue
                            
                            # Extract the additional tech field details
                            tech_field_group_id = patent['cpc_current'][0]['cpc_group_id']
                            tech_field_group = patent['cpc_current'][0].get('cpc_category', 'Unknown')
                            tech_field_subgroup_id = patent['cpc_current'][0].get('cpc_subgroup_id', 'Unknown')
                            tech_field_subgroup = patent['cpc_current'][0].get('cpc_current', 'Unknown')

                            citations = patent['patent_num_times_cited_by_us_patents']
                            abstract = patent['patent_abstract']
                                                        
                            if patent['citedby_patents'][0]['citedby_patent_date'] is not None:
                                citedby_patents = [CitedByPatent(pat['citedby_patent_number'], datetime.strptime(pat['citedby_patent_date'], "%Y-%m-%d")) for pat in patent['citedby_patents']]
                            else:
                                citedby_patents = []  # No citing patents

                            to_add = Patent(
                                patent_id=patent['patent_id'],
                                forward_citations=citations,
                                date_application=patent_date,
                                date_granted=None,  # No granted date here
                                abstract=abstract,
                                tech_field_group=tech_field_group,
                                tech_field_group_id=tech_field_group_id,
                                tech_field_subgroup=tech_field_subgroup,
                                tech_field_subgroup_id=tech_field_subgroup_id,
                                citedby_patents=citedby_patents  # Now includes the adjustment
                            )


                           
                            if patent_date < date:
                                patents_before.append(to_add)
                            else:
                                patents_after.append(to_add)

                    last_patent_id = patents[-1]['patent_id']
                    patent_url = f'{patent_base_url}{{"{assignee_field}": "{closest_match}"}}&f={fields}&o={{"size":1000, "after":"{last_patent_id}"}}&s=[{{"patent_id":"asc"}}]'
                    print("Size of patents before: ", len(patents_before), "Size of patents after: ", len(patents_after))

                else:
                    break
            else:
                print(f"Error: Could not retrieve patent data from ASSIGNEE. Error code: {patent_response.status_code}")
                return None

        return patents_before, patents_after

    # Source: 'legacy'
    elif source == 'legacy':
        patent_base_url = 'https://api.patentsview.org/patents/query?o={"page":1,"per_page":10000}&q='
        assignee_field = 'assignee_organization'
        fields = '["cpc_subgroup_id", "cpc_group_id", "cpc_group_title", "cpc_subgroup_title" ,"app_date", "patent_date","patent_id", "patent_date","cpc_category","assignee_country", "assignee_organization", "assignee_id", "patent_num_cited_by_us_patents", "patent_abstract", "citedby_patent_id", "citedby_patent_title", "citedby_patent_date"]'
        patent_url = f'{patent_base_url}{{"{assignee_field}": "{closest_match}"}}&f={fields}'
        print(patent_url)

        patents_before = []
        patents_after = []
        date = datetime.strptime(date, "%Y-%m-%d")
        last_page = 1

        while True:
            fake_headers = {'User-Agent': ua.random}
            patent_response = requests.get(patent_url, headers=fake_headers)
            
            if patent_response.status_code == 200:
                patent_data = patent_response.json()
                patents = patent_data['patents']
                
                if patents:
                    for patent in patents:
                        if patent['applications'][0]['app_date']:
                            filing_date = datetime.strptime(patent['applications'][0]['app_date'], "%Y-%m-%d")
                            grant_date = datetime.strptime(patent['patent_date'], "%Y-%m-%d")
                            if filing_date.year < first_year:
                                continue
                            
                            ## Tech fieldss
                            tech_field_group_id = patent['cpcs'][0].get('cpc_group_id', "Unknown")
                            tech_field_group = patent['cpcs'][0].get('cpc_category', 'Unknown')
                            tech_field_subgroup_id = patent['cpcs'][0].get('cpc_subgroup_id', 'Unknown')
                            tech_field_subgroup = patent['cpcs'][0].get('cpc_subgroup_title', 'Unknown')

                            citations = patent['patent_num_cited_by_us_patents'] ## citations
                            abstract = patent['patent_abstract'] ## abstracts

                            ## Assignee
                            assignee_organization = patent['assignees'][0].get('assignee_organization', 'Unknown')
                            assignee_key_id = str(patent['assignees'][0].get('assignee_key_id', 'Unknown'))
                            assignee_country = str(patent['assignees'][0].get('assignee_country', 'Unknown'))
                            
                            if patent['citedby_patents'][0]['citedby_patent_date'] is not None:
                                citedby_patents = [CitedByPatent(pat['citedby_patent_id'], datetime.strptime(pat['citedby_patent_date'], "%Y-%m-%d")) for pat in patent['citedby_patents']]
                            else:
                                citedby_patents = None  # No citing patents


                            to_add = Patent(
                                patent_id=patent['patent_id'],
                                forward_citations=citations,
                                date_application=filing_date,
                                date_granted=grant_date,
                                abstract=abstract,
                                tech_field_group=tech_field_group,
                                tech_field_group_id=tech_field_group_id,
                                tech_field_subgroup=tech_field_subgroup,
                                tech_field_subgroup_id=tech_field_subgroup_id,
                                assignee_organization=assignee_organization,
                                assignee_country=assignee_country,
                                assignee_id = assignee_key_id,
                                citedby_patents = citedby_patents
                            )
                           
                            if filing_date < date:
                                patents_before.append(to_add)
                            else:
                                patents_after.append(to_add)
                    
                    last_page += 1
                    patent_url = f'{patent_base_url}&q={{"{assignee_field}": "{closest_match}"}}&f={fields}&o={{"page":{last_page},"per_page":10000}}'
                    print("Size of patents before: ", len(patents_before), "Size of patents after: ", len(patents_after))

                if patent_data['count'] < 10000:
                    break

            else:
                print(f"Error: Could not retrieve patent data from LEGACY. Code: {patent_response.status_code}")
                return None

        return patents_before, patents_after
import random
import time
import requests
from typing import List
from datetime import datetime

def get_patents_from_fields(field, year, group_only=False, partial_call = False, source = "legacy"):
    global scraperapi_key
    #Timer start
    start = time.time()

    year = str(year)
    
    analyzed_fields = {}

    if field not in analyzed_fields:
        analyzed_fields[field] = {}

    if year not in analyzed_fields[field]:
        analyzed_fields[field][year] = {}


    analyzed_fields[field][year]["patents"] = []

        # Source: 'assignee'
    if source == 'assignee':
        patent_base_url = 'https://search.patentsview.org/api/v1/patent/'
        assignee_field = 'assignees.assignee_organization, assignees.assignee_country, assignees.assignee_id'

        if group_only:
            query = f'{{"_and":[{{"_and":[{{"cpc_at_issue.cpc_subclass_id":"{field}"}},{{"cpc_at_issue.cpc_sequence":1}}]}},{{"_gte":{{"application.filing_date":"{year}-01-01"}}}},{{"_lte":{{"application.filing_date":"{year}-12-31"}}}}]}}'
        else:
            query = f'{{"_and":[{{"_and":[{{"cpc_at_issue.cpc_group_id":"{field}"}},{{"cpc_at_issue.cpc_sequence":1}}]}},{{"_gte":{{"application.filing_date":"{year}-01-01"}}}},{{"_lte":{{"application.filing_date":"{year}-12-31"}}}}]}}'
        
        if not partial_call:
            fields = '["cpc_at_issue.cpc_group_id", "cpc_at_issue.cpc_subclass_id", "patent_num_times_cited_by_us_patents","application.filing_date", "patent_id",  "patent_abstract", "assignees.assignee_organization", "assignees.assignee_country", "assignees.assignee_id"]'
        
        else:
            fields = '["patent_id", "patent_abstract"]'
        
        patent_url = f'{patent_base_url}?q={query}&f={fields}'
        print(patent_url)

        while True:
            time.sleep(random.random())
            patent_response = requests.get(patent_url, headers=headers)
            print(patent_url)
            
            if patent_response.status_code == 200:
                patent_data = patent_response.json()
                patents = patent_data['patents']
                
                if patents:
                    for patent in patents:
                        patent_obj = Patent(
                            patent_id=patent['patent_id'],
                            abstract=patent['patent_abstract']
                        )

                        # Append the Patent object to the list
                        analyzed_fields[field][year]["patents"].append(patent_obj)

                    last_patent_id = patents[-1]['patent_id']
                    patent_url = f'{patent_base_url}?q={query}&f={fields}&o={{"size":1000, "after":"{last_patent_id}"}}&s=[{{"patent_id":"asc"}}]'

                else:
                    break
            else:
                print(f"Error: Could not retrieve patent data from ASSIGNEE. Error code: {patent_response.status_code}")
                return None

        ## Timer ends & report progress
        end = time.time()
        print(f"It took {end - start:.2f} sec to retrieve patents under Partial call = {partial_call}")

        count = len(analyzed_fields[field][year]['patents'])  # Total number of patents
        return analyzed_fields, count
    
    else:
        # Construct the query URL
        base_url = 'https://api.patentsview.org/patents/query?o={"page":1,"per_page":10000}&q='

        # Define group specific queries
        if group_only:
            query = f'{{"_and":[{{"_and":[{{"cpc_group_id":"{field}"}},{{"cpc_sequence":0}}]}},{{"_gte":{{"app_date":"{year}-01-01"}}}},{{"_lte":{{"app_date":"{year}-12-31"}}}}]}}'
        else:
            query = f'{{"_and":[{{"_and":[{{"cpc_subgroup_id":"{field}"}},{{"cpc_sequence":0}}]}},{{"_gte":{{"app_date":"{year}-01-01"}}}},{{"_lte":{{"app_date":"{year}-12-31"}}}}]}}'

        
        ## Full search - retrieving all necessary patent data
        if not partial_call:
            ## Change URL based on the field requested
            fields = '["cpc_category", "cpc_subgroup_id", "cpc_group_id", "cpc_group_title", "cpc_subgroup_title", "app_date", "patent_date","patent_id", "patent_abstract", "assignee_country", "assignee_organization", "assignee_id", "patent_num_cited_by_us_patents", "citedby_patent_id", "citedby_patent_title", "citedby_patent_date"]'
                    
        else:
            fields = '["patent_id", "patent_abstract"]'
            
        # Full url
        full_url = f'{base_url}{query}&f={fields}'
        #print(full_url)

        # Counter for last page for moving into next batch of 10k patents
        last_page = 1

        # Function to generate ScraperAPI URL with the target URL
        def get_scraperapi_url(target_url):
            return f"http://api.scraperapi.com?api_key={scraperapi_key}&url={target_url}"


        while True:
            headers = {'User-Agent': ua.random}  # Randomize user-agent to avoid blocking
            tt = random.random()  # Random sleep time between requests
            #print(f'Sleeping for: {tt:.2f} seconds')
            time.sleep(tt + 2)
            #print(full_url)
            
            #headers = {'User-Agent': random.choice(user_agents)}  # Randomize user-agent
            
            full_url = get_scraperapi_url(full_url) 
            response = requests.get(full_url)#, headers=headers)
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
                    
                    # search cpecific data retrieval (we only retrieve ID and Abstract for partial search)
                    if not partial_call:
                        # Create a list of CitedByPatent objects
                        if patent['citedby_patents'][0]['citedby_patent_date'] is not None:
                            citedby_patents = [CitedByPatent(pat['citedby_patent_id'], datetime.strptime(pat['citedby_patent_date'], "%Y-%m-%d")) for pat in patent['citedby_patents']]
                        else:
                            citedby_patents = []  # No citing patents
                        
                        # Fix the dates and format
                        filing_date = datetime.strptime(patent['applications'][0]['app_date'], "%Y-%m-%d")
                        grant_date = datetime.strptime(patent['patent_date'], "%Y-%m-%d")

                        # Fields
                        tech_field_group_id = patent['cpcs'][0]['cpc_group_id']
                        tech_field_group = patent['cpcs'][0].get('cpc_category', 'Unknown')
                        tech_field_subgroup_id = patent['cpcs'][0].get('cpc_subgroup_id', 'Unknown')
                        tech_field_subgroup = patent['cpcs'][0].get('cpc_subgroup_title', 'Unknown')

                        ## Assignee
                        assignee_organization = patent['assignees'][0].get('assignee_organization', 'Unknown')
                        assignee_key_id = str(patent['assignees'][0].get('assignee_key_id', 'Unknown'))
                        assignee_country = str(patent['assignees'][0].get('assignee_country', 'Unknown'))

                
                        # Create the Patent object
                        patent_obj = Patent(
                            patent_id = patent['patent_id'],
                            forward_citations = patent['patent_num_cited_by_us_patents'],
                            date_application = filing_date,
                            date_granted = grant_date,  # Assuming you don't have the granted date from the API
                            abstract = patent['patent_abstract'],
                            tech_field_group=tech_field_group,
                            tech_field_group_id=tech_field_group_id,
                            tech_field_subgroup=tech_field_subgroup,
                            tech_field_subgroup_id=tech_field_subgroup_id, 
                            assignee_organization=assignee_organization,
                            assignee_country=assignee_country,
                            assignee_id = assignee_key_id,
                            citedby_patents = citedby_patents
                        )

                    else:
                        # Create the Patent object
                        patent_obj = Patent(
                            patent_id = patent['patent_id'],
                            abstract = patent['patent_abstract'],
                        )


                    # Append the Patent object to the list
                    analyzed_fields[field][year]["patents"].append(patent_obj)
                    
                # Update the URL for the next page
                half_url = f'https://api.patentsview.org/patents/query?o={{"page":{last_page},"per_page":10000}}&q='
                full_url = f'{half_url}{query}&f={fields}'
                
                if data['count'] < 10000:  # Break the loop if fewer than 10,000 patents are returned
                    break
            else:
                print('Error code is: ', response.status_code)
                analyzed_fields[field][year] = None
                return None

        # timer ends
        end = time.time()
        print(f"It took {end - start:.2f} sec to retrieve patents under Partial call = {partial_call}")
        count = len(analyzed_fields[field][year]['patents'])  # Total number of patents
        return analyzed_fields, count
