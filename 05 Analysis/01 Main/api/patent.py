from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from collections import OrderedDict
import dill as pickle 
import numpy as np
from path_utils import get_base_path
import os 

## Get base path depending on the environment and set path for checked patent (Patent objects) 
base_path = get_base_path()
checked_patents_relative_path = "05 Analysis/01 Main/00 Python data/checked_patents.pkl"
checked_patents_full_path = os.path.join(base_path, checked_patents_relative_path)


@dataclass
class CitedByPatent:
    patent_id: str
    date_granted: datetime

    # Method to calculate the number of days since the patent was granted
    def days_since_granted(self) -> int:
        current_date = datetime.now()
        delta = current_date - self.date_granted
        return delta.days

@dataclass
class Patent:
    patent_id: str
    abstract: str 
    forward_citations: Optional[int] = 0
    date_application: Optional[datetime] = None
    date_granted: Optional[datetime] = None
    tech_field_group: str = ""
    tech_field_group_id: str = ""
    tech_field_subgroup: str = ""
    tech_field_subgroup_id: str = ""
    assignee_organization: str = ""
    assignee_country: str = ""
    assignee_id: str = ""
    citedby_patents: List[CitedByPatent] = field(default_factory=list)

    # Add ClosestPatent attribute of type Optional[Patent]
    closest_patent: Optional['Patent'] = None  # Initialized as None by default

    # Initialize patent_embedding as None, expecting a 1x1024 array later
    patent_embedding: Optional[np.ndarray] = None

    # Initialize distances as none
    euclidean_distance_to_closest_patent : Optional[float] = None
    cosine_similarity_with_closest_patent : Optional[float] = None

    def set_embedding(self, embedding: np.ndarray):
        """Sets the embedding with the expected shape (1, 1024)."""
        if embedding.shape == (1024,):
            self.patent_embedding = embedding
        else:
            raise ValueError(f"Expected embedding of shape (1024,), got {embedding.shape}")
        

    # Method to calculate the total number of cited patents
    def total_cited_patents(self) -> int:
        return len(self.citedby_patents)

    # Method to calculate the total forward citations across all cited by patents
    def total_forward_citations(self) -> int:
        return self.forward_citations

    # Method to add a cited by patent
    def add_cited_by_patent(self, citedby_patent: CitedByPatent):
        self.citedby_patents.append(citedby_patent)

    # Method to count citations over years since application and grant
    def count_citations_by_year(self):
        citations_by_application_years = {}
        citations_by_granted_years = {}

        for cited_patent in self.citedby_patents:
            # Check for years after application date
            years_after_application = cited_patent.date_granted.year - self.date_application.year
            if years_after_application >= 0:  # Only count years after or on the application year
                if years_after_application not in citations_by_application_years:
                    citations_by_application_years[years_after_application] = 0
                citations_by_application_years[years_after_application] += 1

            # Check for years after granted date
            years_after_granted = cited_patent.date_granted.year - self.date_granted.year
            if years_after_granted >= 0:  # Only count years after or on the granted year
                if years_after_granted not in citations_by_granted_years:
                    citations_by_granted_years[years_after_granted] = 0
                citations_by_granted_years[years_after_granted] += 1

        # Sort both dictionaries by key and return as OrderedDicts
        sorted_by_application = OrderedDict(sorted(citations_by_application_years.items()))
        sorted_by_granted = OrderedDict(sorted(citations_by_granted_years.items()))

        return sorted_by_application, sorted_by_granted


import dill as pickle
import os
import time

import os
import pickle
from datetime import datetime

# Function to get the current timestamp as a formatted string
def get_current_timestamp():
    return datetime.now().strftime('%Y.%m.%d_%H:%M')

# Function to get the most recent file
def get_most_recent_file(folder_path, default_file):
    # List all files in the folder that start with 'CheckedPatents_CLSonly' and end with '.pkl'
    files = [f for f in os.listdir(folder_path) if f.startswith('CheckedPatents_CLSonly') and f.endswith('.pkl')]
    
    # If no files are found, use the default file
    if not files:
        print(f"No files found. Using the default file: {default_file}")
        return default_file
    
    # Sort the files by modification time (most recent first)
    files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=True)
    
    # Return the most recent file
    most_recent_file = os.path.join(folder_path, files[0])
    print(f"Loading from most recent file: {most_recent_file}")
    return most_recent_file

# Saving patents to a new file with the current timestamp
def save_patents_with_embeddings(new_patents, checked_patents_folder, default_file):
    """Saves patent_id as the key and patent_embedding as the value to a new file with a timestamp."""
    
    # Get the most recent file or fallback to the default
    checked_patents_full_path = get_most_recent_file(checked_patents_folder, default_file)
    
    # Load existing patents from the pickle file
    try:
        existing_patents = load_patents(checked_patents_full_path)
    except FileNotFoundError:
        print(f"No patent file found at {checked_patents_full_path}. Creating a new one.")
        existing_patents = {}

    # If a single patent is passed, convert it to a list of one element
    #if not isinstance(patent_list, list):
        #patent_list = [patent_list]

    # Convert input list of Patent objects to a dictionary with patent_id as the key
    # and only patent_embedding as the value
    #new_patents = {patent.patent_id: patent.patent_embedding
                   #for patent in patent_list if patent.patent_embedding is not None}

    # Update the existing patents with new ones
    existing_patents.update(new_patents)

    # Generate a new filename with the current timestamp
    timestamp = get_current_timestamp()
    new_file_name = f'CheckedPatents_CLSonly_{timestamp}.pkl'
    new_file_path = os.path.join(checked_patents_folder, new_file_name)

    # Safely save the updated dictionary to the new file using atomic write
    temp_filename = new_file_path + '.tmp'
    with open(temp_filename, 'wb') as f:
        pickle.dump(existing_patents, f)

    # Replace the temporary file with the final file
    os.replace(temp_filename, new_file_path)

    print(f"Patents have been updated and saved to {new_file_path}.")


# Periodic saving function (save patents in batches)
def save_patents_in_batches(patent_list, checked_patents_full_path, batch_size=100):
    """Saves patents in batches to avoid losing all progress if interrupted."""
    try:
        # Load existing patents
        existing_patents = load_patents(checked_patents_full_path)

        # Process patents in batches
        temp_patents = {}
        for i, patent in enumerate(patent_list):
            temp_patents[patent.patent_id] = patent.patent_embedding

            if (i + 1) % batch_size == 0:
                existing_patents.update(temp_patents)
                save_patents_with_embeddings([], checked_patents_full_path)  # Atomic save
                print(f"Saved batch {i // batch_size} at {time.ctime()}")
                temp_patents.clear()

        # Save any remaining patents
        if temp_patents:
            existing_patents.update(temp_patents)
            save_patents_with_embeddings([], checked_patents_full_path)
        
    except KeyboardInterrupt:
        print("Interrupted! Saving current progress before exiting...")
        save_patents_with_embeddings([], checked_patents_full_path)  # Ensure save before exiting

# Loading patents from file
def load_patents(checked_patents_full_path):
    """Loads the dictionary of Patent objects from a file."""
    print('Full path to pickle file: ', checked_patents_full_path)
    try:
        if os.path.exists(checked_patents_full_path) and os.path.getsize(checked_patents_full_path) > 5:  # Check if file exists and is not empty
            with open(checked_patents_full_path, 'rb') as f:
                patent_dict = pickle.load(f)
            return patent_dict
        else:
            return {}
    except FileNotFoundError:
        # Create an empty dictionary or any empty object to pickle
        empty_object = {}

        # Write the empty object to a pickle file
        with open(checked_patents_full_path, 'wb') as file:
            pickle.dump(empty_object, file)
        print(f"No file found: {checked_patents_full_path}. Creating an empty file!!!")
        return {}

# Checking if a patent has been checked
def is_patent_checked(patent_id, checked_patents):
    """Checks if a patent with the given patent_id is in the dictionary."""
    return patent_id in checked_patents
