* ==============================================================================
* A. Set paths
* ==============================================================================
    clear all

    ** set path
    gl analysis "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main"
    gl python "${analysis}/00 Python data"
    gl stata "${analysis}/01 Stata"
    
    gl est "${stata}/01 Main/02 Estimation"
    gl data "${est}/00 Consolidate data"
        gl aux "${data}/_aux"
        gl do  "${data}/do"
        gl dta "${data}/dta"
        gl out "${data}/out"
        gl raw "${data}/raw"
        gl temp "${data}/temp"

* ==============================================================================
* B. Load matched samples with citation counts
* ==============================================================================
    ** Load
    mkf matches 
    cwf matches

    import delimited using "${raw}/working_sampple.csv", clear

    ** Quarters
    replace quarter = substr(quarter, 3, strlen(quarter))
    destring quarter, replace

    ** Drop if both citations are missing
    drop if mi(citations_treated) & mi(citations_control)

    replace citations_control = 0 if mi(citations_control)
    replace citations_treated = 0 if mi(citations_treated)


    ** Rename for reshaping
    rename *_id id*


    rename idmatch match_id

    ** Reshape
    reshape long id citations_, i(lambda quarter match_id) j(treatment) str

    encode treatment, gen(aux)
    drop treatment 
    rename aux treatment
    rename id patent_id
    rename quarter rel_quarter
    rename citations citation

    ** merge with acquired patents data to get relevant columns
    tostring patent_id, replace
    merge m:1 patent_id using "${raw}/04 All patents.dta", gen(merge_match) keep(1 3) // the ones not matched in the using data are the dropped acquisitions that did not make criteria

    ** Drop the 66 controls somehow appearing in the data
    preserve
        keep if treatment == 1 & merge_match == 3 // drop controls who appear in the treated group 
        keep match_id 
        duplicates drop

        tempfile todrop
        save "`todrop'"
    restore

    merge m:1 match_id using "`todrop'", assert(1 3) keep(1) nogen

    assert treatment == 1 if merge_match == 1
    assert treatment == 2 if merge_match == 3
     
********************************************************************************
* C. Sync timing
********************************************************************************

    ** Create quarters
    gen aux = qofd(acq_date)

    bys match_id (aux): carryforward aux, gen(acq_quarter)

    gen quarter = acq_quarter + rel_quarter
    format %tq quarter acq_quarter

    ** Make acq_quarter 0 for controls
    replace acq_quarter = 0 if treatment == 1

    assert !mi(acq_quarter) & !mi(quarter)

    ** Carryforward other cosntant vars
    gsort match_id -ult_parent
    by match_id: carryforward ult_parent acq_type acq_date acq_year deal_id cpc_subclass_current, replace

    ** Make patent_id and quarter unique
    bys lambda patent_id quarter: gen weight = _N

    
    assert weight == 1 if treatment == 2
    assert treatment == 1 if weight > 1
    *duplicates drop lambda patent_id quarter, force
    
********************************************************************************
* D. Save
********************************************************************************
    sort lambda match_id quarter 
    order lambda match_id patent_id ult_parent deal_id treatment quarter acq_quarter rel_quarter acq_type citation *dist* cpc_subclass_current

    compress
    save "${dta}/01 Matched sample with citations.dta", replace