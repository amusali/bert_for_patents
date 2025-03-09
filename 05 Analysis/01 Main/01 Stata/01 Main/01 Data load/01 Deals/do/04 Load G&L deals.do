* ==============================================================================
* A. Set paths
* ==============================================================================
    clear all

    ** set path
    gl analysis "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main"
    gl python "${analysis}/00 Python data"
    gl stata "${analysis}/01 Stata"
    
    gl dl "${stata}/01 Main/01 Data load"
    gl deals "${dl}/01 Deals"
        gl aux "${deals}/_aux"
        gl do  "${deals}/do"
        gl dta "${deals}/dta"
        gl out "${deals}/out"
        gl raw "${deals}/raw"
        gl temp "${deals}/temp"

* ==============================================================================
* B. Load Gauiter & Lamesch data
* ==============================================================================
    ** Load
    import excel using "${raw}/Gautier & Lamesch data on acquisitions.xls", firstrow clear

    ** Handle strings
    qui ds *, has(type string)
    foreach var in `r(varlist)'{
        replace `var' = "" if `var' == "NA"
        destring `var', replace
    }

    ** Harmonize GAFAM names
    replace Acquirer = "AMZ" if Acquirer == "AMZN"
    replace Acquirer = "FB" if Acquirer == "FCBK"

    ** Harmonize varnames
    rename (Companyname Acquirer Acquisitionyear) (child ult_parent acquisition_date)
    order ult_parent child acquisition_date

    ** Save
    duplicates drop child, force
    compress
    save "${dta}/04 G&L deals.dta", replace


