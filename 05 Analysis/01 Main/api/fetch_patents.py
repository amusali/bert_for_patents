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
import random
from numpy.linalg import norm
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf


tf.config.optimizer.set_jit(True)  


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
        fields = '["cpc_subgroup_id","app_date","patent_id", "patent_date","cpc_category","assignee_country", "patent_num_cited_by_us_patents", "patent_abstract", "citedby_patent_id", "citedby_patent_title", "citedby_patent_date"]'

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
                            tech_field = patent['cpcs'][0]['cpc_subgroup_id']
                            abstract = patent['patent_abstract']
                            citedby_patents = patent['citedby_patents']
                            patent_year = patent_date.year
                            to_add = {'patent_id':patent['patent_id'], 'citations': citations, 'tech_field': tech_field, 'year': patent_date, 'abstract': abstract, 'citedby_patents' : citedby_patents}  
                           
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

    

def get_patents_from_fields(field, year):
    year = str(year)
    
    analyzed_fields = {}

    if field not in analyzed_fields:
        analyzed_fields[field] = {}

    if str(year) not in analyzed_fields[field]:
        analyzed_fields[field][year] = {}
   
        base_url = 'https://api.patentsview.org/patents/query?o={"page":1,"per_page":10000}&q='
        query = f'{{"_and":[{{"_and":[{{"cpc_subgroup_id":"{field}"}},{{"cpc_sequence":0}}]}},{{"_gte":{{"app_date":"{year}-01-01"}}}},{{"_lte":{{"app_date":"{year}-12-31"}}}}]}}'
        fields = '["patent_num_cited_by_us_patents", "patent_id", "patent_abstract", "assignee_organization"]'
        full_url = f'{base_url}{query}&f={fields}'

        last_page = 1

        analyzed_fields[field][year]["patents"] = []

        while True:
            
            headers = {'User-Agent': ua.random}

            tt = random.random()
            print('Sleeping for: ', tt, 'seconds')
            time.sleep(tt)
            
            response = requests.get(full_url, headers = headers)
            if response.status_code == 200:
                last_page += 1
                data = response.json()
                patents = data['patents']
                if not patents:
                    print('NO PATENTS HERE')
                    analyzed_fields[field][year] = None
                    return None
                for patent in patents:
                       analyzed_fields[field][year]["patents"].append(
                                                              {"patent_id": patent['patent_id'],
                                                               "abstract" : patent['patent_abstract'],
                                                               "assignee_organization" : patent['assignees'][0]['assignee_organization']}
                                                               )
                half_url = f'https://api.patentsview.org/patents/query?o={{"page":{last_page},"per_page":10000}}&q='
                full_url = f'{half_url}{query}&f={fields}'
                
                if data['count'] < 10000:
                    break
            else:
                print('Error code is: ', response.status_code)
                analyzed_fields[field][year] = None
                return None        
        return analyzed_fields, data['total_patent_count']

from api.bertembeddings import get_embd_of_whole_abstract
import tensorflow as tf
import numpy as np

def get_embeddings_from_field(patent,
                            filter_tfidf,
                            assignee_file=r"C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main\api\treated_assignees.xlsx",
                            ):
    """
    Input: A patent which is dictionary with following keys:
        - patent_id - str
        - tech_field - str
        - year - datetime.datetime object
        - abstract - str
    
    Output: Returns embeddings of every patent in the field of a patent - in list
    """

    ## Get patent data
    target_field = patent['tech_field']
    year = patent['year'].year
    target_patent_id = patent['patent_id']

    ## Get embedding of all the patents in the same CPC sub-group

    ### Find the abstracts in the same CPC sub-group
    resp = get_patents_from_fields(target_field, year)
    patents_to_compare = resp[0][target_field][str(year)]['patents']
    total_count = resp[1]
    print('There are in total', total_count-1, 'patents to be compared against.')

    ### Eliminate treated patents
    try: ## Read the file
        df = pd.read_excel(assignee_file)
    except FileNotFoundError:
        print(f"The file {assignee_file} was not found in the project folder.")
        return False
    
    #### Check whether the assignees are treated or not 
    filtered_patents = [d for d in patents_to_compare if d.get('assignee_organization') not in df['Assignees'].values] 

    ### Remove target patent and get abstracts
    abstracts_to_compare = [d['abstract'] for d in filtered_patents if d.get('patent_id') != target_patent_id]

    ### Eliminate unlikely similar patents using TF-IDF
    if filter_tfidf:

        tfidf_distances, indx = get_tfidf_from_field(abstracts_to_compare, patent['abstract'])
        filtered_abstracts = filter_patents_by_similarity(abstracts_to_compare, tfidf_distances, indx)

        ### Report on the progress
        print(f"There were {total_count - 1 } files in total. After TF-IDF filtering, there are {len(filtered_abstracts)} patents left now.")
    else:
        filtered_abstracts = abstracts_to_compare

    ### Get embeddings of abstracts in the same CPC sub-group
    docs_embeddings = []
    for abs in filtered_abstracts:
        docs_embeddings.append(get_embd_of_whole_abstract(abs, has_context_token=True))

    return docs_embeddings, filtered_abstracts



def get_embedding_of_target_and_field(patent, filter_tfidf = True):
    """
    Input: A patent which is dictionary with following keys:
        - patent_id - str
        - tech_field - str
        - year - datetime.datetime object
        - abstract - str
    
    Output: returns two things:
        i) List of embeddings from the field of target patent
        ii) An array of embedding of target patent

    """
     
    ## Get abstract embeddings to compare against
    embd_of_to_compare_against, patents_to_compare_against = get_embeddings_from_field(patent, filter_tfidf)

    ## Get own abstract embedding
    embd_of_patent_being_compared = get_embd_of_whole_abstract(patent['abstract'], has_context_token=True)

    return  embd_of_patent_being_compared, embd_of_to_compare_against, patents_to_compare_against

def find_the_closest_abstract_excerpt(patent, filter_tfidf):
    own, against, patents = get_embedding_of_target_and_field(patent, filter_tfidf)
    dist_eu, index_eu, dist_cs, index_cs = find_distances(own, against)
    
    if index_cs != index_eu:
        print("The closest abstract is different across metrics")
        return None
    else:
        print(f"Cosine similarity is {dist_cs[index_cs]}")
        return [patent['abstract'], patents[index_cs]]
    

def find_distances(embd_of_patent_being_compared, embd_of_to_compare_against):
    """
    Input: Two inputs
        i) An array of embedding of the target patent
        ii) A list of embeddings of patents within the tech field of target patent

    Output: returns the distance according to given metric and the index of closest patent
    """

    ## Get abstract embeddings to compare against
    n = len(embd_of_to_compare_against) # number of abstracts being compared against

    ### Initialize a distance matrix to store distances
    distances_euclidean = np.zeros(n)
    
    ### Euclidean distance
        # Compute the Euclidean distance between each pair of arrays
    for i in range(n):
        distances_euclidean[i] = np.linalg.norm(embd_of_to_compare_against[i] - embd_of_patent_being_compared)
        closest_patent_euclidean_index = np.argmin(distances_euclidean)

    ### Cosine similarity
    ### Compute cosine similarity between each pair of arrays
    distances_cs = cosine_similarity([embd_of_patent_being_compared], embd_of_to_compare_against).flatten()
    closest_patent_cs_index = np.argmax(distances_cs)
    
    return distances_euclidean, closest_patent_euclidean_index, distances_cs, closest_patent_cs_index

def get_tfidf_from_field(abstracts, target_abstract):

    """
    Input: Two inputs:
        i) abstracts - a list of abstracts to be compared against (containing strings)
        ii) target_abstract - a string of target patent's abstract
    
    Output: returns the distance according to given metric and the index of closest patent
    """

    # Report
    print("Currently doing TF-IDF")
    
    # Append target patent
    abstracts.append(target_abstract)

    # Step 1: Compute TF-IDF vectors for all abstracts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(abstracts)

    # Step 2: Compute cosine similarity between a given patent and all others
    target_idx = len(abstracts) - 1  # Index of the patent you're comparing
    cosine_similarities = cosine_similarity(tfidf_matrix[target_idx], tfidf_matrix)

    return cosine_similarities, target_idx

def filter_patents_by_similarity(abstracts, cosine_similarities, target_idx, threshold=0.1):
    """
    Filters the list of abstracts based on cosine similarity.
    
    Input:
        i) abstracts - the list of patent abstracts
        ii) cosine_similarities - the 2D array of cosine similarity values (output from get_tfidf_from_field)
        iii) target_idx - the index of the target patent (output from get_tfidf_from_field)
        iv) threshold - the cosine similarity threshold for filtering (default is 0.1)
    
    Output:
        - A new list of abstracts with the last patent removed and those below the threshold filtered out.
    """

    # Step 1: Remove the last abstract (the target patent)
    filtered_abstracts = abstracts[:target_idx]  # Exclude the last patent (target_idx)

    # Step 2: Extract cosine similarity values for the other patents
    # The similarities array is 2D, so we get the first row
    similarities = cosine_similarities[0][:target_idx]

    # Step 3: Find the indices where the cosine similarity is below the threshold
    indices_below_threshold = [i for i, similarity in enumerate(similarities) if similarity < threshold]

    # Step 4: Filter the abstracts based on the indices below the threshold
    final_abstracts = [abstract for i, abstract in enumerate(filtered_abstracts) if i not in indices_below_threshold]

    return final_abstracts

            
            


# Function to get the maximum off-diagonal entry for a specific row
def max_off_diagonal(df, row_index):
    """
    Input : similarity of a DataFrame with dimension [length of ]
    
    """
    row = df.loc[row_index]  # Get the specific row
    # Exclude the diagonal entry (where row_index == column_index)
    off_diagonal = row.drop(df.columns[row_index])  # Drop the diagonal element
    max_column = off_diagonal.idxmax()
    min_column = off_diagonal.idxmin()
    max_value = off_diagonal.max()  # Find the max of remaining entries
    return max_value, max_column, min_column

"""

#%% Intensity Function        
def find_patent_intensity(name, date,source = None,  assignee_id = None):
    before = []
    after = []
    if assignee_id is not None:
        result = get_patents_from_matches(assignee_id, date)
        print('Getting it from ID')
        if result is not None:
            before.append(result[0])
            after.append(result[1])
        before = [item for subitem in before for item in subitem]
        after = [item for subitem in after for item in subitem]
        x = before
        y = after
        if (not x and not y) or (x==0 and y==0):
            print('Not found patents either before  after')
            return None
    else:    
        res = get_patents(str(name), date, source)
        if res is None:
            return None
        else:
            x = res[0]
            y = res[1]
        if (not x and not y) or (x==0 and y==0):
            print('Not found patents either before  after')
            return None
    count = 0
    uzun_x = len(x)
    for patent in x:
        count += 1
        if patent['tech_field'] is None:
            continue
        
        
        average = average_citations_finder(patent['tech_field'], patent['year'].year)
        if average is None or average == 0:
            #print('NONE or zero AVERAGE')
            continue
        
        if patent['citations'] == 0:
            intensity = 0
        else:           
            intensity = int(patent['citations'])/float(average)
        patent['intensity'] = intensity
       # if count % 250 == 0:
           # print('Found intensities for', count, 'patents from x', uzun_x - count, 'patents left')
            
    count = 0
    print('RETURNED ALL PATENTS BEFORE')
    uzun_y = len(y)
    for patent in y:
        count += 1
        if patent['tech_field'] is None:
            continue  
        average = average_citations_finder(patent['tech_field'], patent['year'].year)
        if average is None or average == 0:
            #print('NONE or Zero AVERAGE')
            continue
        
        if patent['citations'] == 0:
            intensity = 0
        else:            
            intensity = int(patent['citations'])/float(average)
        patent['intensity'] = intensity
       # if count % 250 == 0:
           # print('Found intensities for', count, 'patents from y', uzun_y - count, 'patents left')
            
    print("RETURNED ALL PATENTS AFTER")
    if assignee_id is not None:
        return x, y
    else:        
        return x, y

"""
