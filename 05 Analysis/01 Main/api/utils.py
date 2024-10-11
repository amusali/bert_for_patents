import re

def clean_text(s):
    # Remove anything inside parentheses along with the parentheses themselves
    s = re.sub(r'\(.*?\)', '', s)

    # Split the modified string into words
    words = re.findall(r'[\w\.]+|[,\)]', s)

    cleaned_words = []
    for word in words:
        # Keep the word as is if it looks like a URL, otherwise remove unwanted characters
        if re.match(r'\b\w+\.\w+\b', word):
            word = re.sub(r'\.$', '', word)  # Remove trailing period
            cleaned_words.append(word)
        else:
            word = word.replace(",", "").replace(".", "")
            cleaned_words.append(word)
    
    return ' '.join(' '.join(cleaned_words).split())

def remove_common_endings(company_name):
    endings = ['svcs', 'inc', 'inc.', 'corp', 'corp.', 'comp', 'research',
               'ltd', 'na', 'lp', 'partners', 'co', 'corporation', 
               'llc', 'partners', ',inc', '.inc', 'inc-', 'consul', 'international',
               'hldg', 'holding', 'services', 'stores', 'trust', 'group', 'systems', 'advisors',
               'technologies', 'fund', 'solutions', 'acq', 'sub', 'grp','holdings', 'mgmt', 'acquisition', 'acquisitions']  # Add more common endings as needed
    unique_names = []
    lowercase_name = company_name.lower()
    
    
    stripped_name = lowercase_name.strip()
    
    while stripped_name.split()[-1] in endings:
        stripped_name = ' '.join(stripped_name.split()[:-1]).strip()
    unique_names.append(stripped_name)
    #unique_names.append(lowercase_name.strip())
    return unique_names
