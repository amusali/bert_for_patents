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
    import excel using "${raw}/Aguirre (2024) data on AI acquisitions.xlsx", firstrow sheet("Merger, Acquistions, and Other ") clear

    ** Rename vars
    dropmiss, force
    dropmiss, obs force
    rename (Company1 Company2) (ult_parent child)

    ** Filter
    keep if Typeofoperation == "Full acquisition"
    keep if inlist(ult_parent, "Alphabet", "Amazon", "Apple", "Meta", "Microsoft")

    ** Harmoinze gafam names
    replace ult_parent = "GOOG" if ult_parent == "Alphabet"
    replace ult_parent = "AMZ" if ult_parent == "Amazon"
    replace ult_parent = "APPL" if ult_parent == "Apple"
    replace ult_parent = "FB" if ult_parent == "Meta"
    replace ult_parent = "MSFT" if ult_parent == "Microsoft"

    ** Date
    destring Year, replace
    gen date_str = Month  + " 1 " + string(Year) if !mi(Month)
    gen acquisition_date = date(date_str, "MDY")
    format acquisition_date %td
    
    ** value
    replace Value = "" if regexm(Value, "disclose") | regexm(Value, "~") | regexm(Value, "rox")
    destring Value, replace
    rename Value deal_value
    rename Notes deal_synopsis

    ** relevant vars
    keeporder ult_parent child acquisition_date deal_value deal_synopsis
    gisid child 
    compress 
    save "${dta}/05 Aguirre deals.dta", replace