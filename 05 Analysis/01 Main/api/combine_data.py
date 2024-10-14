def combine_sdc_truematches():

    import pandas as pd
    from  path_utils import get_base_path
    import os

    # Set path for the M&A data
    base = get_base_path()
    relative =  r"05 Analysis\01 Main\01 Stata\raw\sdc_gafam\SDC Platinum sdc_gafam_usonly_completed.xlsx"
    sdc_path = os.path.join(base, relative)

    # Load the Excel file for M&A data
    sdc = pd.read_excel(sdc_path, sheet_name="Request 7", header=2)

    # Set path for Assignees data checked by Google
    relative = r"05 Analysis\01 Main\00 Python data\True Matches by Google.xlsx"
    true_matches_path = os.path.join(base, relative)

    # Load the Excel file for True matches by google
    true_mathces = pd.read_excel(true_matches_path)

    # Merge them 
    merged_df = pd.merge(sdc, true_mathces, left_on="Target Full Name", right_on="Target Company", how="inner")

    return merged_df
