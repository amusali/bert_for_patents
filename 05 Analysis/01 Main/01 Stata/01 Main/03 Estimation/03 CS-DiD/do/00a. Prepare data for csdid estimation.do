* ==============================================================================
* A. Set paths
* ==============================================================================
    clear all
    capture log close
    set more off
    set type double, perm
    set excelxlsxlargefile on

    ** set path
    gl analysis "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main"
    gl python "${analysis}/00 Python data"
    gl stata "${analysis}/01 Stata"
    
    gl est "${stata}/01 Main/03 Estimation"
    gl csdid "${est}/03 CS-DiD"
        gl aux "${csdid}/_aux"
        gl do  "${csdid}/do"
        gl dta "${csdid}/dta"
        gl out "${csdid}/out"
        gl raw "${csdid}/raw"
        gl temp "${csdid}/temp"

    gl raw_drive "G:\My Drive\PhD Data\12 Sample Final\actual results\citation"

    ** Locals
    local list_of_maximum_periods 12 16 20 40 // quarters (i.e. 3, 4, 5, 10 years)
    local seed = 1709
    local B = 100 // number of bootstrap replications

* ==============================================================================
* B. Load all the IDs from matched records
* ==============================================================================

    ** Load 
    u "${raw_drive}/00 Matched IDs - no pair info.dta", clear

    ** Bring filename
    merge m:1 file_id using "${raw_drive}/00 Matched IDs - file map.dta", assert(3) nogen
    drop path
    rename filename config
    replace config = strtrim(subinstr(config, "01 Hybrid matches -", "", .))

    ** Rename and merge with patents metadata
    rename id patent_id
    merge m:1 patent_id using "G:\My Drive\PhD Data\09 Acquired patents\04 All patents.dta", assert(1 2 3) keep(1 3) gen(merge_csdid)

    ** Drop patents that are used as controls but are indeed acquired as part of a deal but not yet greated, these are patents that are never used in matching and samples and need to be dropped
    drop if treated == 0 & merge_csdid == 3  // drops 88k observations

    *** NOw assers
    assert merge_csdid == 3 if treated == 1
    assert merge_csdid == 1 if treated == 0
    tab merge_csdid treated
    drop merge_csdid

    assert acq_type == "M&A" | acq_type == "" if regexm(config, "M&A")
    assert regexm(config, "M&A") if acq_type == "M&A"

    assert acq_type == "Off deal" | acq_type == "" if regexm(config, "Off deal")
    assert regexm(config, "Off deal") if acq_type == "Off deal"

    ** Keep relevant vars
    keeporder lambda_val patent_id treated config ult_parent acq_date grant_date deal_id assignee num_claims cpc_group_at_issue cpc_group_current cpc_subclass_current cpc_subclass_at_issue
    compress
    duplicates drop

    ** Gen quarter variable understood by Stata
    gen grant_quarter = qofd(grant_date)
    format grant_quarter %tq

    ** Get grant date of controls
    mkf patents
    cwf patents 
     
    u "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main\01 Stata\01 Main\01 Data load\00 Patents\dta\01 Patent data - without citations.dta", clear

    ** Filter and make Grant date a Stata quarter date
    keep patent_id patent_date cpc_subclass_current
    duplicates drop 
    gisid patent_id

    gen grant_date = date(patent_date, "YMD")
    format grant_date %td
    drop patent_date

    gen grant_quarter = qofd(grant_date)
    format grant_quarter %tq

    gen grant_year = yofd(grant_date)

    tempfile patents
    save "`patents'"
    
    ** Merge back into main data
    cwf default 

    merge m:1 patent_id using "`patents'", update replace assert(2 3 4) keep(3 4) nogen 

    assert !mi(grant_date) & !mi(grant_quarter)

    ** Save intermediary
    compress 
    gisid lambda_val patent_id config

    save "${dta}\00 Sample - without citations.dta", replace