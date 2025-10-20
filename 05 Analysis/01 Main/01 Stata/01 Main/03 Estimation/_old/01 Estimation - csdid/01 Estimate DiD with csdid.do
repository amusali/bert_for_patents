
* ==============================================================================
* A. Set paths
* ==============================================================================

    ** set path
    gl analysis "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main"
    gl python "${analysis}/00 Python data"
    gl stata "${analysis}/01 Stata"
    
    gl csdid "${stata}/02 Estimation/01 Estimation - csdid"
        gl do "${csdid}/do"
        gl dta "${csdid}/dta"
        gl out "${csdid}/out"
        gl raw "${csdid}/raw"

* ==============================================================================
* B. Load Patent data & clean
* ==============================================================================
    
    ** Load data
    clear all
    import excel using "${raw}/patents_to_quarterly_df.xlsx", firstrow clear

    ** Clean dates
    gen quarter = qofd(dofc(quarter_date))
    gen grant = qofd(dofc(grant_date))
    gen acquisition_date = qofd(dofc(acquired_date))
    gen year = yofd(dofc)
    format quarter grant acquisition_date %tq 

    drop grant_date quarter_date acquired_date
    rename grant grant_date

    ** Generate cohort variable - G 
    bys patent_id: egen aux = sum(treatment) 
    gen never_treated = aux == 0

    gen cohort = 0 if never_treated
    replace cohort = acquisition_date if !never_treated
    format %tq cohort

    ** Drop duplicates based on patent_id and quarter
    drop A match *dist* acquisition_date quarters cosine
    duplicates drop  

    ** Drop duplicates based on patent_id and acquisition date
    /* 
    This is because a patent that has been treated once always should stay
    treated according to the staggered timing of treatment in csdid
    */
    preserve
        keep patent_id cohort

        duplicates drop
        duplicates tag patent_id, gen(dup)
        gen tag = 1 if dup > 0

        drop dup cohort
        duplicates drop

        tempfile to_drop
        save "`to_drop'"
    restore

    *** Bring back the duplicates and drop the patents that have been acquired more than once
    merge m:1 patent_id using "`to_drop'", assert(1 3) nogen
    drop if tag == 1

    ** Drop the patents that are split into two different spelling of the same assignees
    **duplicates drop patent_id assignee_organization, force 
    
    gisid patent_id quarter 

    ** Set xtset
    sort patent_id quarter
    egen patent_id_group = group(patent_id)
    xtset patent_id_group quarter


* ==============================================================================
gen year = yofd(dofq(quarter))
gen g = yofd(dofq(cohort))

collapse (sum) forward, by(patent_id year)

