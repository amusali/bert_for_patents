import api.bertembeddings as be
import tensorflow as tf
import numpy as np
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from api.fetch_patents import get_patents_from_fields, get_quasipatents_from_field

import api.patent as apipat
import api.fetch_patents
import importlib
importlib.reload(apipat)
from datetime import datetime
from path_utils import get_base_path
tf.config.optimizer.set_jit(True)  

## Get base path depending on the environment
base_path = get_base_path()


# Check if running in Colab or locally for checked patents file
if 'COLAB_GPU' in os.environ:
    checked_patents_full_path = '/content/drive/MyDrive/PhD Data/01 CLS Embeddings/CheckedPatents_CLSonly_2024.10.20_23.28.pkl'
    #if not os.path.exists(checked_patents_full_path):
    #    os.makedirs(checked_patents_full_path)
else:
    checked_patents_relative_path = "05 Analysis/01 Main/00 Python data/CheckedPatents_CLSonly.pkl"
    checked_patents_full_path = os.path.join(base_path, checked_patents_relative_path)

# Assignee file
assignee_file = os.path.join(base_path, "05 Analysis/01 Main/00 Python data/True Matches by Google.xlsx")
try: ## Read the file of treated assignees
    df = pd.read_excel(assignee_file)
except FileNotFoundError:
    print(f"The file {assignee_file} was not found in the project folder.")
    #return False

# Global storage for newly computed embeddings and fields
new_patent_embeddings = {}
new_fields = {}

# Initialize the global variable to None
checked_patents = None
field_dict = None

# Function to get the current timestamp as a formatted string
def get_current_timestamp():
    return datetime.now().strftime('%Y.%m.%d_%H:%M')

# Save embeddings to a new file every 20 minutes
def save_patents_every_20_minutes(checked_patents_folder, default_file):

    # Save the current state of embeddings to a new file
    print(f"Saving new embeddings")
    apipat.save_patents_with_embeddings(new_patent_embeddings, checked_patents_folder,  default_file)
    
    # Clear the memory dictionary after saving
    new_patent_embeddings.clear()

    # Reset the global checked_patents variable so it will reload next time
    global checked_patents
    checked_patents = None  # Set it to None to ensure it gets reloaded next time

    print("Checked patents data has been reset. It will be reloaded next time.")


# Save embeddings to a new file every 20 minutes
def save_field_dict_every_20_minutes(field_dict_folder, default_file):

    # Save the current state of embeddings to a new file
    print(f"Saving new embeddings")
    api.fetch_patents.save_field_dict(new_fields, field_dict_folder, default_file)
    
    # Clear the memory dictionary after saving
    new_fields.clear()

    # Reset the global checked_patents variable so it will reload next time
    global field_dict
    field_dict = None  # Set it to None to ensure it gets reloaded next time

    print("Field dict data has been reset. It will be reloaded next time.")

# Load the most recent file based on the timestamp
def load_most_recent_checked_patents(checked_patents_folder):
    files = [f for f in os.listdir(checked_patents_folder) if f.startswith('Field dict - quasi patents') and f.endswith('.pkl')]
    if not files:
        return {}
    
    # Sort files by modified time and load the most recent one
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(checked_patents_folder, f)))
    latest_file_path = os.path.join(checked_patents_folder, latest_file)
    
    print(f"Loading checked patents from {latest_file}...")
    return apipat.load_patents(latest_file_path)

def load_most_recent_field_dict(field_dict_folder):
    files = [f for f in os.listdir(field_dict_folder) if f.startswith('Field dict - quasi patents') and f.endswith('.pkl')]
    if not files:
        return {}
    
    # Sort files by modified time and load the most recent one
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(field_dict_folder, f)))
    latest_file_path = os.path.join(field_dict_folder, latest_file)
    
    print(f"Loading field dict from {latest_file}...")
    return api.fetch_patents.load_field_dict(latest_file_path)


def get_embeddings_from_field(patent,
                            group_only,
                            filter_tfidf = True,
                            batch_size = 32,
                            df = df,
                            checked_patents_file=checked_patents_full_path,
                            search_threshold = 1000,
                            ):
    """
    Args:
            i) Patent object see api.patent.py
            ii) Group Only - True if the search should be broad (over group), o/w False
            iii) Batch size 
            iv) Filter TF-IDF - True by default

    Output: Returns embeddings of every patent in the field of a patent - in list
    """
    # gLOBALS
    global new_patent_embeddings
    global new_fields
    global checked_patents
    global field_dict 
      # Load the most recent checked patents file at the start
    if checked_patents is None:
        checked_patents = load_most_recent_checked_patents('/content/drive/MyDrive/PhD Data/01 CLS Embeddings')
    if field_dict is None:
        field_dict = load_most_recent_field_dict('/content/drive/MyDrive/PhD Data/04 Field dictionaries')

    ## Get patent data
    if group_only:    
        target_field = patent.tech_field_group_id

    else:
        target_field = patent.tech_field_subgroup_id

    if target_field is None:
        print('NULL tech field of the given patent')
        return None, None
    
    year = patent.date_application.year
    target_patent_id = patent.patent_id

    ## Retrieve quasi-patents to check whether the field-year has been loaded before
    if target_field in field_dict:
        if year in field_dict[target_field]:
            print("Loading patents from Field Dictionary")
            patents_to_compare = field_dict[target_field][str(year)]
        else:
            patents_to_compare = get_quasipatents_from_field(target_field, year, group_only)[0]
            if patents_to_compare is None:
                return None, None
            if target_field in new_fields:
                new_fields[target_field][str(year)] = patents_to_compare
            else:
                new_fields[target_field] = {}
                new_fields[target_field][str(year)] = patents_to_compare
    else:
        patents_to_compare = get_quasipatents_from_field(target_field, year, group_only)[0]
        if patents_to_compare is None:
                return None, None
        if target_field in new_fields:
            new_fields[target_field][str(year)] = patents_to_compare
        else:
            new_fields[target_field] = {}
            new_fields[target_field][str(year)] = patents_to_compare
    print(patents_to_compare)

    ## Get embedding of all the patents in the same CPC sub-group

    ## Find the abstracts in the same CPC sub-group in the same APPLICATION YEAR
    
    #resp = get_patents_from_fields(target_field, year, group_only)
    #resp = get_quasipatents_from_field(target_field, year, group_only)
    #if resp is None:
        #return None, None
    #patents_to_compare = resp[0][target_field][str(year)]['patents']
    
    print('There are in total', len(patents_to_compare), 'patents to be compared against.')

    ### Eliminate treated patents
        #### Filter out patents that are treated or the target patent itself
    filtered_patents = [d for d in patents_to_compare if d.assignee_organization not in df['Assignees'].values and d.patent_id != target_patent_id] 

    ### Case of no filtered patents (e.g. G06N3/0455 in 2017 only contains WAVEONE INC. patents )
    if filtered_patents == []: ## NEED TO DEAL WITH LATER
        return None, None
        get_embeddings_from_field(patent, group_only=True, search_threshold=5)
        
    ### Eliminate unlikely similar patents using TF-IDF
    if filter_tfidf:
        if len(filtered_patents) > search_threshold:
            filtered_patents = filter_patents_by_tfidf(filtered_patents, patent)

            ### Report on the progress
            print(f"There were {len(patents_to_compare)} files in total. After TF-IDF filtering, there are {len(filtered_patents)} patents left now.")

    ## Prepare list to store embeddings in the original order
    docs_embeddings = []

    ## Batching
    filtered_abstracts = [pat.abstract for pat in filtered_patents]
    batched_abstracts = [filtered_abstracts[i:i + batch_size] for i in range(0, len(filtered_abstracts), batch_size)]

    
    ## Iterate over filtered_patents in their original order
    for filtered_patent in filtered_patents:
        if apipat.is_patent_checked(filtered_patent.patent_id, checked_patents):
            # Retrieve the embeddings of already checked patents
            if checked_patents[filtered_patent.patent_id] is not None:
                #print(f"Patent {filtered_patent.patent_id} has been processed before. Retrieving embeddings...")
                #filtered_patent.set_embedding(checked_patents[filtered_patent.patent_id])
                #print("adding embedding")
                docs_embeddings.append(checked_patents[filtered_patent.patent_id])
        else:
            # Add to a list of abstracts for those that need embeddings computed
            docs_embeddings.append(None)  # Placeholder for embeddings to be computed

    ## Compute embeddings for those that need them (embeddings marked as None)
    abstracts_to_compute = [filtered_patents[i].abstract for i, emb in enumerate(docs_embeddings) if emb is None]
    if abstracts_to_compute:
        batched_abstracts = [abstracts_to_compute[i:i + batch_size] for i in range(0, len(abstracts_to_compute), batch_size)]

        computed_embeddings = []
        for batch in batched_abstracts:
            computed_embeddings.append(be.get_embd_of_whole_abstract(batch, has_context_token=True))
        computed_embeddings = np.vstack(computed_embeddings)  # Vertically stack embeddings

        ## Assign the computed embeddings back to their placeholders
        computed_idx = 0
        for i in range(len(docs_embeddings)):
            if docs_embeddings[i] is None:
                # Update the patent and save it to the checked patents dictionary
                filtered_patents[i].set_embedding(computed_embeddings[computed_idx])
                docs_embeddings[i] = computed_embeddings[computed_idx]

                # Add new embeddings to the temporary dictionary for later saving
                new_patent_embeddings[filtered_patents[i].patent_id] = filtered_patents[i]
                computed_idx += 1

    # Save the updated checked_patents back to the pickle file
    #apipat.save_patents_with_embeddings(filtered_patents, checked_patents_full_path=checked_patents_file)
    #print(docs_embeddings)

    return np.vstack(docs_embeddings), filtered_patents


def get_embedding_of_target_and_field(patent, group_only, batch_size, filter_tfidf = True):
    """
    Input: i) Parent object
            ii) Group Only - True if the search should be broad (over group), o/w False
            iii) Batch size 
            iv) Filter TF-IDF - True by default

    
    Output: returns two things:
        i) List of embeddings from the field of target patent
        ii) An array of embedding of target patent

    """
     
    ## Get abstract embeddings to compare against
    

    embd_of_to_compare_against,  patents_to_compare_against = get_embeddings_from_field(patent, group_only, filter_tfidf, batch_size)
    if embd_of_to_compare_against is None:
        return None, None, None
   

    ## Get own abstract embedding
    embd_of_patent_being_compared = be.get_embd_of_whole_abstract(patent.abstract, has_context_token=True)
    patent.patent_embedding = embd_of_patent_being_compared

    return  embd_of_patent_being_compared, embd_of_to_compare_against, patents_to_compare_against
    
def find_closest_patent(patent, group_only, batch_size,  filter_tfidf, metric = 'cosinesim'):
    """
    Args: i) Patent object
          ii) Group only - True if broad tech field search, o/w False
          iii) Batch size
          iv) Filter TF-IDF indicator
          v) Metric - 'cosinesim' by default, the only other option is euclidean metric.
    
    """
    ## Get own and against embeddings
    
    own, against, patents = get_embedding_of_target_and_field(patent, group_only, batch_size, filter_tfidf)
    if own is None:
        return None, None, None
    
    dist_eu, index_eu, dist_cs, index_cs = find_distances(own, against)

    ## Create dict to return
    if metric == "cosinesim":
        indx = index_cs
    else:
        indx = index_eu
    
    return patents[indx], dist_cs[indx], dist_eu[indx] 
    
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
    distances_cs = cosine_similarity(embd_of_patent_being_compared, embd_of_to_compare_against).flatten()
    closest_patent_cs_index = np.argmax(distances_cs)
    
    return distances_euclidean, closest_patent_euclidean_index, distances_cs, closest_patent_cs_index

def filter_patents_by_tfidf(patents, target_patent, threshold = 0.1):

    """
    Input: Two inputs:
        i) Patents to compare against (Patent objects)
        ii) Target patent (Patent object)
    
    Output: returns the distance according to given metric and the index of target patent
    """

    # Report
    print("Currently doing TF-IDF")
    
    # Get abstracts 
    abstracts = [patent.abstract for patent in patents]

    # Append target patent abstract
    abstracts.append(target_patent.abstract)

    # Step 1: Compute TF-IDF vectors for all abstracts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(abstracts)

    # Step 2: Compute cosine similarity between a given patent and all others
    target_idx = len(abstracts)-1 # Index of the patent you're comparing
    cosine_similarities = cosine_similarity(tfidf_matrix[target_idx], tfidf_matrix)

    # Step 2: Extract cosine similarity values for the other patents
    # The similarities array is 2D, so we get the first row
    similarities = cosine_similarities[0][:-1]

    ## Size check
    #print(similarities)
    #print(similarities.shape)
    assert similarities.shape[0] == len(patents), f"Cosine similarity array size {similarities.shape} does not match the size of ToComparePatents with size: {len(patents)}"

    # Step 3: Find the indices where the cosine similarity is below the threshold
    indices_below_threshold = [i for i, similarity in enumerate(similarities) if similarity < threshold]

    # Step 4: Filter the abstracts based on the indices below the threshold
    final_patents = [patent for i, patent in enumerate(patents) if i not in indices_below_threshold]

    return final_patents
