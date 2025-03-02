
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

* ==============================================================================
* B. Load Patent data & combine with CPC at Issue
* ==============================================================================

    * Load Patent Data
    mkf patent 
    cwf patent 
    
    import delimited using "${raw}/g_patent.tsv"

    * Load CPC at Issue
    mkf cpc_at_issue
    cwf cpc_at_issue 
    
    import delimited using "${raw}/g_cpc_at_issue.tsv"

    ** Take the fisrt assigned CPC
    keep if cpc_sequence == 1
    drop cpc_sequence

    tempfile cpc_at_issue
    save "`cpc_at_issue'"

    ** Merge Patent Data with CPC at Issue
    cwf patent
    merge 1:1 patent_id using `cpc_at_issue', assert(1 3) gen(merge_cpc_at_issue)

    ** Rename cpc vars
    foreach var in cpc* {
        rename `var' `var'_at_issue
    }

********************************************************************************
* C. Combine with Current CPC 
********************************************************************************
    mkf cpc_current
    cwf cpc_current

    ** Load Current CPC
    import delimited using "${raw}/g_cpc_current.tsv"

    ** Take the fisrt assigned CPC
    keep if cpc_seq == 0

    ** Rename cpc vars
    foreach var in cpc* {
        rename `var' `var'_current
    }

    ** Make Patent ID string for merging with patent data
    tostring patent_id, replace

    tempfile cpc_current
    save "`cpc_current'"

    ** Merge Patent Data with the current CPC
    cwf patent
    merge 1:1 patent_id using `cpc_current', assert(1 3) gen(merge_cpc_current)


********************************************************************************
* D. Combine with Patent abstracts 
********************************************************************************

    mkf abstracts
    cwf abstracts   

    ** Load Patent Abstracts
    import delimited using "${raw}/g_patent_abstract.tsv"

    drop if _n == 1
    rename (v1 v2) (patent_id abstract)

    tempfile abstract
    save "`abstract'"

    ** Merge Patent Data with the abstracts
    cwf patent
    merge 1:1 patent_id using `abstract', assert(3) nogen


********************************************************************************
* E. Combine with Assigne Data
********************************************************************************
    *---------------------------------------------------------*
    * E.1. Disambiguated Assignee Data
    *---------------------------------------------------------*
        ** Load
        mkf assignee_disamb
        cwf assignee_disamb

        import delimited using "${raw}/g_assignee_disambiguated.tsv", clear

        ** Drop inventor names
        keep patent_id disambig_assignee_organization assignee_sequence
        drop if mi(disambig_assignee_organization)

        ** Create an indicator for multiplee-assignee patents
        duplicates tag patent_id, gen(dup)
        gen has_multiple_assignees = dup > 0
        drop dup

        sort patent_id assignee_sequence

        ** Combine info for multiple assinges
        reshape wide disambig_assignee_organization, i(patent_id) j(assignee_sequence)
        gisid patent_id
        rename disambig_assignee_organization* assignee*

        ** Rename assignee vars
        qui ds patent_id, not
        foreach var in `r(varlist)' {
            rename `var' `var'_disamb
        }

        tempfile assignee_disamb
        save "`assignee_disamb'"

    *---------------------------------------------------------*
    * E.2. Undisambiguated Assignee Data
    *---------------------------------------------------------*
        ** Load
        mkf assignee_undisamb
        cwf assignee_undisamb

        import delimited using "${raw}/g_assignee_not_disambiguated.tsv"

        ** Clean
        keep patent_id assignee_sequence raw_assignee_organization
        drop if mi(raw_assignee_organization)

        gsort patent_id assignee_sequence

        ** Combine info for multiple assinges
        reshape wide raw_assignee_organization, i(patent_id) j(assignee_sequence)
        gisid patent_id
        rename raw_assignee_organization* assignee*

        ** Rename assignee vars
        qui ds patent_id, not
        foreach var in `r(varlist)' {
            rename `var' `var'_notdisamb
        }

        tempfile assignee_not_disamb
        save "`assignee_not_disamb'"


    *---------------------------------------------------------*
    * E.3. Merge assignee vars
    *---------------------------------------------------------*
    cwf patent
    merge 1:1 patent_id using `assignee_disamb', assert(1 3) gen(merge_assignee_disamb)
    merge 1:1 patent_id using `assignee_not_disamb', assert(1 3) gen(merge_assignee_notdisamb)


    compress 
    save "${dta}/01 Patent data - without citations.dta", replace