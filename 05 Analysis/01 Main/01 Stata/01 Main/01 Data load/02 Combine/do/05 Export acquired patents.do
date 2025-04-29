* ==============================================================================
* A. Set paths
* ==============================================================================
    clear all

    ** set path
    gl analysis "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main"
    gl python "${analysis}/00 Python data"
    gl stata "${analysis}/01 Stata"
    
    gl dl "${stata}/01 Main/01 Data load"
    gl deals "${dl}/02 Combine"
        gl aux "${deals}/_aux"
        gl do  "${deals}/do"
        gl dta "${deals}/dta"
        gl out "${deals}/out"
        gl raw "${deals}/raw"
        gl temp "${deals}/temp"

* ==============================================================================
* B. Load identified patents data
* ==============================================================================
    ** Load
    use "${dta}/04 All patents.dta", clear	

    assert !mi(acquired)
    assert !mi(acq_date) & !mi(grant_date)
    keep if acquired
    drop if mi(cpc_subclass_current)
    keep patent_id acq_date grant_date cpc_subclass_current

    export delimited using "${out}/acquired_patents.csv", replace