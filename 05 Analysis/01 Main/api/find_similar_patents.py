import api.bertembeddings as be
import tensorflow as tf
import numpy as np
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from api.fetch_patents import get_patents_from_fields

import api.patent as apipat
import importlib
importlib.reload(apipat)

from path_utils import get_base_path
tf.config.optimizer.set_jit(True)  

## Get base path depending on the environment
base_path = get_base_path()
checked_patents_relative_path = "05 Analysis/01 Main/00 Python data/checked_patents.pkl"
checked_patents_full_path = os.path.join(base_path, checked_patents_relative_path)

def get_embeddings_from_field(patent,
                            group_only,
                            filter_tfidf,
                            batch_size = 32,
                            assignee_file = os.path.join(base_path, "05 Analysis/01 Main/api/treated_assignees.xlsx"),
                            checked_patents_file=checked_patents_full_path,
                            ):
    """
    Args:
            i) Patent object see api.patent.py
            ii) Group Only - True if the search should be broad (over group), o/w False
            iii) Batch size 
            iv) Filter TF-IDF - True by default

    Output: Returns embeddings of every patent in the field of a patent - in list
    """

    # Load the checked patents from the pkl file
    checked_patents = apipat.load_patents(checked_patents_file)

    ## Get patent data
    if group_only:    
        target_field = patent.tech_field_group_id
    else:
        target_field = patent.tech_field_subgroup_id

    year = patent.date_application.year
    target_patent_id = patent.patent_id

    ## Get embedding of all the patents in the same CPC sub-group

    ### Find the abstracts in the same CPC sub-group in the same APPLICATION YEAR
    resp = get_patents_from_fields(target_field, year, group_only)
    patents_to_compare = resp[0][target_field][str(year)]['patents']
    #total_count = resp[1]
    print('There are in total', len(patents_to_compare), 'patents to be compared against.')

    ### Eliminate treated patents
    try: ## Read the file
        df = pd.read_excel(assignee_file)
    except FileNotFoundError:
        print(f"The file {assignee_file} was not found in the project folder.")
        return False
    
    #### Filter out patents that are treated or the target patent itself
    filtered_patents = [d for d in patents_to_compare if patent.assignee_organization not in df['Assignees'].values and d.patent_id != target_patent_id] 

    ### Eliminate unlikely similar patents using TF-IDF
    if filter_tfidf:

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
            if checked_patents[filtered_patent.patent_id].patent_embedding is not None:
                print(f"Patent {filtered_patent.patent_id} has been processed before. Retrieving embeddings...")
                filtered_patent.set_embedding(checked_patents[filtered_patent.patent_id].patent_embedding)
                docs_embeddings.append(filtered_patent.patent_embedding)
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
                checked_patents[filtered_patents[i].patent_id] = filtered_patents[i]
                computed_idx += 1

    # Save the updated checked_patents back to the pickle file
    apipat.save_patents(list(checked_patents.values()))

    return np.vstack(docs_embeddings), filtered_abstracts, filtered_patents


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
    embd_of_to_compare_against, abstracts_to_compare_against, patents_to_compare_against = get_embeddings_from_field(patent, group_only, filter_tfidf, batch_size)

    ## Get own abstract embedding
    embd_of_patent_being_compared = be.get_embd_of_whole_abstract(patent.abstract, has_context_token=True)
    patent.patent_embedding = embd_of_patent_being_compared

    return  embd_of_patent_being_compared, embd_of_to_compare_against, abstracts_to_compare_against, patents_to_compare_against
    
def find_closest_patent(patent, group_only, batch_size,  filter_tfidf, metric = 'cosinesim'):
    """
    Args: i) Patent object
          ii) Group only - True if broad tech field search, o/w False
          iii) Batch size
          iv) Filter TF-IDF indicator
          v) Metric - 'cosinesim' by default, the only other option is euclidean metric.
    
    """
    ## Get own and against embeddings
    own, against, abstracts, patents = get_embedding_of_target_and_field(patent, group_only, batch_size, filter_tfidf)
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
