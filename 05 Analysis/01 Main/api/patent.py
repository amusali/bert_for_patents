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
from collections import OrderedDict

@dataclass
class Patent:
    patent_id: str
    forward_citations: int
    date_application: datetime
    date_granted: datetime
    abstract: str
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


# Saving patents to file
def save_patents(patent_list, checked_patents_full_path=checked_patents_full_path):
    """Adds new Patent objects to the existing checked_patents file without overwriting."""
    
    # Load existing patents from the pickle file
    try:
        existing_patents = load_patents(checked_patents_full_path)
    except FileNotFoundError:
        print(f"No existing patent file found at {checked_patents_full_path}. Creating a new one.")
        existing_patents = {}
    
    # If a single abstract (string) is passed, convert it to a list of one element
    if not isinstance(patent_list, list):
        patent_list = [patent_list]

    # Convert input list of Patent objects to a dictionary with patent_id as the key
    new_patents = {patent.patent_id: patent for patent in patent_list}

    # Update the existing patents with new patents
    existing_patents.update(new_patents)

    # Save the updated dictionary back to the file
    with open(checked_patents_full_path, 'wb') as f:
        pickle.dump(existing_patents, f)
    
    print(f"Patents have been updated and saved to {checked_patents_full_path}.")


# Loading patents from file
def load_patents(checked_patents_full_path = checked_patents_full_path):
    """Loads the dictionary of Patent objects from a file."""
    try:
        with open(checked_patents_full_path, 'rb') as f:
            patent_dict = pickle.load(f)
        return patent_dict
    except FileNotFoundError:
        print(f"No file found: {checked_patents_full_path}")
        return {}


# Checking if a patent has been checked
def is_patent_checked(patent_id, checked_patents):
    """Checks if a patent with the given patent_id is in the dictionary."""
    return patent_id in checked_patents
