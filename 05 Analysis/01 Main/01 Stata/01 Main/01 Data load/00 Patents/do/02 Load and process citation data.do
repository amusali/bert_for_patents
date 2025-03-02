
* ==============================================================================
* A. Set paths
* ==============================================================================

    clear all 

    ** Set paths
    gl analysis "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main"
    gl python "${analysis}/00 Python data"
    gl stata "${analysis}/01 Stata"
    
    gl dl "${stata}/01 Main/01 Data load"
        gl do "${dl}/do"
        gl dta "${dl}/dta"
        gl out "${dl}/out"
        gl raw "${dl}/raw"

* ==============================================================================
* B. Load Citation data & combine with patents data
* ==============================================================================

    *---------------------------------------------------------*
    * B.1. Combine two datasets
    *---------------------------------------------------------*
        * Load Citation Data    
        import delimited using "${raw}/g_us_patent_citation.tsv", clear colrange(:3) varnames(1)
        describe

        ** Combine with patents data
        mkf patents
        cwf patents 

        ** Load patents data
        import delimited using "${raw}/g_patent.tsv", clear colrange(:3) varnames(1)

        tempfile patents
        save "`patents'"

        cwf default
        merge m:1 patent_id using `patents', assert(2 3) keep(3) nogen

    *---------------------------------------------------------*
    * B.2. Clean and save
    *---------------------------------------------------------*
        ** Drop irrelevant vars
        keep patent_id citation_patent_id patent_date

        ** Rename
        rename (patent_id citation_patent_id patent_date) (citedby_patent_id patent_id citation_date)

        ** date
        gen aux = date(citation_date, "YMD")
        drop citation_date
        rename aux citation_date
        format %td citation_date

        ** Save
        compress 
        save "${dta}/02 Patent citations - raw.dta", replace

    