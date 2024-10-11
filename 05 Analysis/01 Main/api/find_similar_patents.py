import api.bertembeddings as be
import tensorflow as tf
import numpy as np
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from api.fetch_patents import get_patents_from_fields

from path_utils import get_base_path
tf.config.optimizer.set_jit(True)  

## Get base path depending on the environment
base_path = get_base_path()

def get_embeddings_from_field(patent,
                            filter_tfidf,
                            group_only,
                            assignee_file = os.path.join(base_path, "05 Analysis/01 Main/api/treated_assignees.xlsx" ),
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
    if group_only:    
        target_field = patent['tech_field_group']
    else:
        target_field = patent['tech_field_subgroup']
    year = patent['year'].year
    target_patent_id = patent['patent_id']

    ## Get embedding of all the patents in the same CPC sub-group

    ### Find the abstracts in the same CPC sub-group
    resp = get_patents_from_fields(target_field, year, group_only)
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
        docs_embeddings.append(be.get_embd_of_whole_abstract(abs, has_context_token=True))

    return docs_embeddings, filtered_abstracts



def get_embedding_of_target_and_field(patent, group_only, filter_tfidf = True):
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
    embd_of_to_compare_against, patents_to_compare_against = get_embeddings_from_field(patent, filter_tfidf, group_only)

    ## Get own abstract embedding
    embd_of_patent_being_compared = be.get_embd_of_whole_abstract(patent['abstract'], has_context_token=True)

    return  embd_of_patent_being_compared, embd_of_to_compare_against, patents_to_compare_against

def find_the_closest_abstract_excerpt(patent, group_only, filter_tfidf):
    own, against, patents = get_embedding_of_target_and_field(patent, group_only, filter_tfidf)
    dist_eu, index_eu, dist_cs, index_cs = find_distances(own, against)
    
    if index_cs != index_eu:

        print("The closest abstract is different across metrics")
        print(f"Cosine similarity is: {dist_cs[index_cs]}")
        return patent['abstract'], patents[index_cs]
    else:
        print(f"Cosine similarity is: {dist_cs[index_cs]}")
        return patent['abstract'], patents[index_cs]
    

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
    
    ### Progress
    print("Size of own", np.shape(embd_of_patent_being_compared))
    print("Size of against", np.shape(embd_of_to_compare_against))
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
