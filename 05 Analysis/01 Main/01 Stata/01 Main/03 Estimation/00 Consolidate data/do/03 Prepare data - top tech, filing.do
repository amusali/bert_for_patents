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

    import delimited using "${raw}/02 Working sample - top tech, filing(1).csv", clear

    ** Quarters
    replace quarter = substr(quarter, 3, strlen(quarter))
    destring quarter, replace

    ** Drop if both citations are missing
    drop if mi(citations_treated) & mi(citations_control)

    replace citations_control = 0 if mi(citations_control)
    replace citations_treated = 0 if mi(citations_treated)

    ** merge with acquired patents data to get relevant columns
    rename treated_id patent_id
    tostring patent_id, replace
    merge m:1 patent_id using "${raw}/04 All patents.dta", nogen assert(2 3) keep(3)

    compress 
    save "${dta}/03 Data for matching estimators - top tech, filing.dta", replace