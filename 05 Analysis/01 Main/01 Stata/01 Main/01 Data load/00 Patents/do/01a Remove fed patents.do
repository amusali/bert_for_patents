* ==============================================================================
* A. Set paths
* ==============================================================================

    ** set path
    gl analysis "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main"
    gl python "${analysis}/00 Python data"
    gl stata "${analysis}/01 Stata"
    
    gl dl "${stata}/01 Main/01 Data load"
        gl do "${dl}/do"
        gl dta "${dl}/dta"
        gl out "${dl}/out"
        gl raw "${dl}/raw"
/*
* ==============================================================================
* B. Load Patent IDs that have been fed to BERT 
* ==============================================================================
    * Load
    clear all
    import delimited using "${raw}/fed_patents.csv"

    rename v1 patent_id 
    tostring patent_id, replace

    tempfile fed_patents
    save "`fed_patents'"

    * Load combined patend data
    use "${dta}/01 Patent data - without citations.dta", clear

    * Clean
    gen grant_date = date(patent_date, "YMD")
    drop if grant_date < td(1jan1990)
    drop if mi(cpc_subclass_at_issue) & mi(cpc_subclass_current)

    merge 1:1 patent_id using `fed_patents', assert(1 3) gen(merge_fed_patents)

********************************************************************************
* C. Find the ordering of CPC subsclasses
********************************************************************************

    preserve
        keep if merge_fed_patents == 3

        * Find the most frequent subclasses
        bys cpc_subclass_current: gen count = _N

        keep count cpc_subclass_current
        duplicates drop
        drop if mi(cpc_subclass_current)

        * Sort by count (highest first)
        gsort -count 
        drop count 
        gen sort_var = _n

        * Save
        tempfile cpcs_ordered
        save "`cpcs_ordered'"
    restore


    * Merge with main data
    merge m:1 cpc_subclass_current using `cpcs_ordered'

    * Sort accourding to cbc_subclass_current
    sort sort_var 

********************************************************************************
* D. Keep the patents to be fed to BERT
********************************************************************************

    * Filter
    keep if merge_fed_patents == 1
    keep patent_id abstract

    * Export
    compress
    export delimited using "${out}/Patents to be fed to BERT.csv", replace