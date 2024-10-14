import os
import dill as pickle

# Define paths
SAVE_DIR = '/content/drive/My Drive/PhD Data/subgroup_checked'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Function to save a list of Patent objects to Drive
def save_patents_to_drive(patents, assignee_name, date_effective):
    filename = f'{assignee_name}_{date_effective}_patents.pkl'
    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(patents, f)

# Function to load Patent objects from Drive
def load_patents_from_drive(assignee_name, date_effective):
    filename = f'{assignee_name}_{date_effective}_patents.pkl'
    filepath = os.path.join(SAVE_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        return None

# Main process to run for each row of the dataframe
def process_assignees(df, get_patents_function, find_closest_patent_function):
    for index, row in df.iterrows():
        assignee_name = row['Assignees']
        date_effective = row['Date Effective']

        # Try to load previously saved patents
        patents = load_patents_from_drive(assignee_name, date_effective)
        if patents is None:
            # Get patents if not previously saved
            patents_before_after = get_patents_function({assignee_name}, {date_effective})
            patents_before = patents_before_after[0]
            patents_after = patents_before_after[1]
            
            # Save patents for future use
            save_patents_to_drive(patents_before, assignee_name, f'{date_effective}_before')
            save_patents_to_drive(patents_after, assignee_name, f'{date_effective}_after')
        else:
            print(f"Patents for {assignee_name} at {date_effective} loaded from Drive.")
            patents_before = patents['before']
            patents_after = patents['after']

        # Now find the closest patent for each in patents_after
        for patent in patents_before:
            closest_patent, distance_cs, distance_eu = find_closest_patent_function(patent, group_only=False, batch_size=32, filter_tfidf=True)
            
            # Add closest patent information to the patent object
            patent.closest_patent = closest_patent
            patent.distance_cs = distance_cs
            patent.distance_eu = distance_eu
        
        # Save updated patent information
        save_patents_to_drive(patents_after, assignee_name, f'{date_effective}_after')


