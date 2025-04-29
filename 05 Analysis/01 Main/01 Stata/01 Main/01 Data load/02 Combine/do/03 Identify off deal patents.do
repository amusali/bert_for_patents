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
    local file = "${analysis}\03 Python scripts\02 Find off deal patents\out\all_patents_transferred_20250311_143020.csv"
    import delimited using "`file'", varnames(1) clear
    
    ** Clean
    duplicates drop patentnumber, force
    drop if patentnumber == "NULL"
    gisid patentnumber
    rename patentnumber patent_id

    ** Harmonize names
    replace assignor = "MOTOROLA" if regexm(assignor, "MOTO")
    replace assignor = "IBM" if regexm(assignor, "INTERNATIONAL")
    replace assignee = "AOL" if regexm(assignee, "AOL")
    replace assignee = "KODAK" if regexm(assignee, "KODAK")
    replace assignee = "MSFT" if regexm(assignee, "MICROSOFT")

    replace assignee = "FB" if regexm(assignee, "FACEBOOK")
    replace assignee = "APPL" if regexm(assignee, "APPLE")
    replace assignee = "GOOG" if regexm(assignee, "GOOGLE")
    replace assignee = "MSFT" if regexm(assignee, "MICROSOFT")
    rename assignee ult_parent

    ** Acquisition date
    gen aux = substr(executiondate, 1, 10)
    gen acquisition_date = date(aux, "YMD")
    drop executiondate aux
    format acquisition_date %td
    gen acquisition_year = yofd(acquisition_date)

    ** Related deal ID
    gen deal_id = 288 if regexm(ult_parent, "NUANCE")
    replace deal_id = 136 if regexm(ult_parent, "MOTOROLA")

    assert acquisition_year < 2022 if regexm(ult_parent, "NUANCE")
    replace acquisition_date = td(03mar2022) if regexm(ult_parent, "NUANCE")

    ** Drop MSFT as assignor // 7 patents to FB only
    drop if assignor == "MICROSOFT CORPORATION"

    preserve
        frame copy default counts
        cwf counts
        drop patent_id
        gen count = _n

        collapse (count) count, by(ult_parent assignor acquisition_date)
        cwf default
    restore

    tempfile offdeal
    save "`offdeal'"

********************************************************************************
* C. Combine with general Patents data
********************************************************************************

    ** Load & merge
    use "${dl}/00 Patents/dta/01 Patent data - without citations.dta", clear
    gen grant_date = date(patent_date, "YMD")
    format grant_date %td
    
    merge 1:1 patent_id using "`offdeal'", keep(3) nogen

    gisid patent_id
    dropmiss, force
    compress
    save "${dta}/03 Acquired patents - off deal.dta", replace
