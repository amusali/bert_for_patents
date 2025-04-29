* ==============================================================================
* A. Set paths
* ==============================================================================
    clear all

    ** set path
    gl analysis "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main"
    gl python "${analysis}/00 Python data"
    gl stata "${analysis}/01 Stata"
    
    gl est "${stata}/01 Main/02 Estimation"
    gl twfe "${est}/01 DiD - TWFE"
        gl aux "${twfe}/_aux"
        gl do  "${twfe}/do"
        gl dta "${twfe}/dta"
        gl out "${twfe}/out"
        gl raw "${twfe}/raw"
        gl temp "${twfe}/temp"

* ==============================================================================
* B. Load matched samples with records
* ==============================================================================
    ** Load 
    import delimited using "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main\03 Python scripts\03 Estimation\02 Top Tech\02 Off deal - closer patents\raw\10 Sample - top tech, Off deal, 2y, filing date.csv", clear 

    rename *_id id_*

    reshape long id_ log_citations_ unique_cpc_ unique_assignee_, i(id_match quarter) j(treat) str
    tab treat

    keep if inlist(treat, "treated", "control")

    cap drop acq_quarter
    gen acq_quarter = qofd(date(acq_date, "YMD"))
    gen q_real = quarter + acq_quarter
    format acq_quarter q_real %tq


    gen treatment = q_real >= acq_quarter & treat == "treated" 


    keep treatment q_real  id_ id_match acq_quarter quarter treat  log_citations_ ult_parent unique_cpc_ unique_assignee_ maha
    rename *_ *
    ren id patent_id


    duplicates tag patent_id q_real, gen(dup)
    tab dup 
    duplicates drop patent_id q_real, force
    
    
    xtset patent_id q_real

    xthdidregress twfe (log_cit) (treatment) if maha < 5, group(patent)



