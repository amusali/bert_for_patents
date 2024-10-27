#@title utils
from api.patent import Patent
import dill as pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

def load(path):
    with open(path, mode = "rb") as f:
        file = pickle.load(f)
    return file
def save(path, file):
    with open(path, mode = "wb") as f:
        pickle.dump(file, f)




def get_assignee_name(filename):
  """
  Extracts the assignee name from the filename.

  Args:
    filename: The filename to extract the assignee name from.

  Returns:
    The assignee name, or None if it could not be extracted.
  """
  match = re.search(r"^(.*?)_", filename)  # Matches everything before the first underscore
  if match:
    return match.group(1)  # Returns the matched group
  else:
    return None  # Returns None if no match is found

def extract_date(filename):
  """Extracts the date in YYYY-MM-DD format from the filename."""
  match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)  # Regular expression to find the date pattern
  if match:
    effective_date = match.group(1)  # Get the matched date string
    return effective_date
  else:
    return None  # Return None if no date is found

def update_patent_groups(old_patents, new_patents):
    """Compares two lists of patents and updates tech_field_group, subgroup,
    tech_field_group_id, and subgroup_id if different in the new list.

    Args:
        old_patents: A list of Patent objects.
        new_patents: A list of Patent objects with potentially updated group information.
    """
    old_patent_dict = {patent.patent_id: patent for patent in old_patents}

    changed_indices = []

    for i, new_patent in enumerate(new_patents):
        if new_patent.patent_id in old_patent_dict:
            old_patent = old_patent_dict[new_patent.patent_id]
            old_patent_index = old_patents.index(old_patent)  # Get the index of the old patent in the original list

            # Update tech_field_group and tech_field_group_id
            if old_patent.tech_field_group != new_patent.tech_field_group or \
               old_patent.tech_field_group_id != new_patent.tech_field_group_id:
                old_patent.tech_field_group = new_patent.tech_field_group
                old_patent.tech_field_group_id = new_patent.tech_field_group_id
                print(f"Updated tech_field_group for patent {new_patent.patent_id}")
                changed_indices.append(old_patent_index)

            # Update subgroup and subgroup_id
            if old_patent.tech_field_subgroup != new_patent.tech_field_subgroup or \
               old_patent.tech_field_subgroup_id != new_patent.tech_field_subgroup_id:
                old_patent.tech_field_subgroup = new_patent.tech_field_subgroup
                old_patent.tech_field_subgroup_id = new_patent.tech_field_subgroup_id
                print(f"Updated subgroup for patent {new_patent.patent_id}")

    return changed_indices

def check_regular_patent_content(patent):
    """Checks if a patent object has all required data.
    Raises AssertionError with a descriptive message if data is missing.

    Args:
        patent: A Patent object.
    """
    try:
        assert patent.closest_patent is not None, f"Patent {patent.patent_id}: closest_patent is missing"
        assert patent.cosine_similarity_with_closest_patent is not None, f"Patent {patent.patent_id}: cosine_similarity_with_closest_patent is missing"
        assert patent.euclidean_distance_to_closest_patent is not None, f"Patent {patent.patent_id}: euclidean_distance_to_closest_patent is missing"
        assert patent.patent_id is not None, f"Patent: patent_id is missing"
        assert patent.abstract is not None, f"Patent {patent.patent_id}: abstract is missing"
        assert patent.date_application is not None, f"Patent {patent.patent_id}: date_application is missing"
        assert patent.patent_embedding is not None, f"Patent {patent.patent_id}: patent_embedding is missing"
        assert patent.tech_field_group is not None, f"Patent {patent.patent_id}: tech_field_group is missing"
        assert patent.tech_field_group_id is not None, f"Patent {patent.patent_id}: tech_field_group_id is missing"
        assert patent.tech_field_subgroup is not None, f"Patent {patent.patent_id}: tech_field_subgroup is missing"
    except AssertionError as e:
        print(e)  # Print the error message
        # You can add other error handling logic here, like logging or raising a custom exception


from typing import List
from scipy.spatial.distance import cosine, euclidean
import numpy as np
from api.bertembeddings import get_embd_of_whole_abstract

def regularize_patents(patents: List[Patent]):
    skipped_patents = []
    
    for i, patent in enumerate(patents):
        
        #print(f"Processing patent {patent.patent_id}")
        # Step 1: Skip patents with no closest_patent and log their IDs
        if patent.closest_patent is None:
            print(f"Skipping patent {patent.patent_id} due to missing closest_patent")
            skipped_patents.append(patent.patent_id)
            continue  # Skip to the next patent

        # Step 2: Check if both patents have embeddings, generate if not
        if patent.patent_embedding is None:
            print(f"Generating embedding for patent {patent.patent_id}")
            patent.set_embedding(get_embd_of_whole_abstract(patent.abstract, has_context_token=True)[-1])

        if patent.closest_patent.patent_embedding is None:
            print(f"Generating embedding for closest patent of patent {patent.patent_id}")
            patent.closest_patent.set_embedding(get_embd_of_whole_abstract(patent.closest_patent.abstract, has_context_token=True)[-1])
        
        # Step2.5: Reshape the embeddings if they are not 1D arrays
        if patent.patent_embedding.shape == (1, 1024):
            patent.patent_embedding = patent.patent_embedding[-1]
        if patent.closest_patent.patent_embedding.shape  == (1, 1024):
            patent.closest_patent.patent_embedding = patent.closest_patent.patent_embedding[-1]        
        # Step 3: Compute or validate the cosine similarity and Euclidean distance
        # Extract embeddings for easy access
        embedding1 = patent.patent_embedding
        embedding2 = patent.closest_patent.patent_embedding

        # Calculate cosine similarity and Euclidean distance if needed
        calculated_cosine_similarity = 1 - cosine(embedding1, embedding2)
        print(calculated_cosine_similarity)
        calculated_euclidean_distance = euclidean(embedding1, embedding2)
        
        # Check and set cosine similarity
        if patent.cosine_similarity_with_closest_patent is None:
            patent.cosine_similarity_with_closest_patent = calculated_cosine_similarity
        else:
            # Verify the stored value matches the calculated one
            assert abs(patent.cosine_similarity_with_closest_patent - calculated_cosine_similarity) < 10**-2, \
                f"Mismatch in cosine similarity for patent {patent.patent_id}: existing: {patent.cosine_similarity_with_closest_patent}, calculated: {calculated_cosine_similarity}"

        # Check and set Euclidean distance
        if patent.euclidean_distance_to_closest_patent is None:
            patent.euclidean_distance_to_closest_patent = calculated_euclidean_distance
        else:
            # Verify the stored value matches the calculated one
            assert abs(patent.euclidean_distance_to_closest_patent - calculated_euclidean_distance) < 10**-2, \
                f"Mismatch in Euclidean distance for patent {patent.patent_id}: existing: {patent.euclidean_distance_to_closest_patent}, calculated: {calculated_euclidean_distance}"

        # Step 4: Final assertion to ensure all required attributes are available
        assert patent.closest_patent is not None, f"Missing closest_patent for {patent.patent_id}"
        assert patent.patent_embedding is not None and patent.closest_patent.patent_embedding is not None, \
            f"Missing embeddings for patent {patent.patent_id} or its closest"
        assert patent.cosine_similarity_with_closest_patent is not None, \
            f"Missing cosine similarity for patent {patent.patent_id}"
        assert patent.euclidean_distance_to_closest_patent is not None, \
            f"Missing Euclidean distance for patent {patent.patent_id}"
        assert all([patent.patent_id, patent.abstract, patent.date_application, patent.date_granted]), \
            f"Incomplete data for patent {patent.patent_id}"
        assert all([patent.closest_patent.patent_id, patent.closest_patent.abstract, 
                    patent.closest_patent.date_application, patent.closest_patent.date_granted]), \
            f"Incomplete data for closest patent of {patent.patent_id}"
    
        print(f"Processed patent {i}/{len(patents)}")
    # Log the IDs of skipped patents
    print("Skipped patents (no closest_patent):", skipped_patents)
    return patents, skipped_patents
