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
* B. Load Patent data
* ==============================================================================
    ** Load
    use "${dl}/00 Patents/dta/01 Patent data - without citations.dta", clear
    
    ** Date
    gen grant_date = date(patent_date, "YMD")
    format grant_date %td

    ** Merge with acquired assignees dataset
    gen tass = assignee0_disamb
    rename tass assignee 
    drop if mi(assignee)

    merge m:1 assignee using "${dta}\01 Acquires assignees.dta", assert(1 3) keep(3) nogen
    
    assert !mi(grant_date) & !mi(acquisition_date)

    ** Drop waymo - wrong deal (it is a project of GOOG)
    drop if assignee == "Waymo LLC"

    ** Save matched patents
    dropmiss, force
    order assignee grant_date acquisition_date patent_id
    sort assignee grant_date acquisition_date patent_id  
    gisid patent_id
    compress
    save "${dta}\02 Acquired patents - through deals, preprocessed.dta", replace