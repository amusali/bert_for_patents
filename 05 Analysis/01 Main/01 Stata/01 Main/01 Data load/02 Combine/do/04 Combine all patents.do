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
    use "${dta}/02 Acquired patents - through deals, preprocessed.dta", clear

    tempfile ma
    save "`ma'"

    use "${dta}/03 Acquired patents - off deal.dta", clear

    tempfile offdeal
    save "`offdeal'"

    *---------------------------------------------------------*
    * B.1. Case of Motorola patents
    *---------------------------------------------------------*
        preserve
            keep if assignor == "MOTOROLA" & acquisition_date < td(1nov2014) // 27 patents bought from Motorola after it was resold to Lenovo

            tempfile motorola
            save "`motorola'"
        restore

        use `ma', clear

        ** Drop Motorola Inc. patents
        keep if deal_id == 136 // motorola deal id
        *drop if assignee == "Motorola, Inc." // not Motorola Mobility!

        
        merge 1:1 patent_id using "`motorola'", keep(2 3)

        ** Patents bought as a result of Motorola deal
        gen modnote = "Bought as a result of Motorola deal" if _merge == 2 & grant_date < td(22may2012) 
        replace modnote = "Developed during the Google time" if _merge == 2 & grant_date >= td(22may2012) & grant_date <acquisition_date

        replace modnote = "Bought as a result of Motorola deal" if _merge == 3 & grant_date <= td(22may2012) 
        replace modnote = "Developed during the Google time" if _merge == 3 & grant_date > td(22may2012)  & grant_date < td(1nov2014)

        drop if mi(modnote)
        drop _merge

        ** Generate acquired indicator
        gen acquired = modnote == "Bought as a result of Motorola deal"

        replace acquisition_date = td(22may2012) if modnote == "Bought as a result of Motorola deal" // Motorola deal date
        replace resold_date = . // since none was resold to lenovo
        replace deal_id = 136
        drop assignor

        tempfile motorola_clean
        save "`motorola_clean'"

    *---------------------------------------------------------*
    * B.2. Case of Nuance patents
    *---------------------------------------------------------*
        use `offdeal', clear
        preserve
            keep if regexm(ult_parent, "NUANCE")

            tempfile nuance
            save "`nuance'"
        restore

        use `ma', clear

        keep if deal_id == 288 // nuance deal id

        append using "`nuance'"

        ** Make patent ID unique
        duplicates drop patent_id, force

        ** Clean
        drop assignor
        replace ult_parent = "MSFT"
        replace acquisition_year = 2022
        gen acquired = 1

        tempfile nuance_clean
        save "`nuance_clean'"

********************************************************************************
* C. Off deal patents
********************************************************************************
    use `offdeal', clear

    drop if assignor == "MOTOROLA"
    drop if regexm(ult_parent, "NUANCE")

    gen acq_type = "Off deal"
    gen acquired = 1

    tempfile offdeal_clean
    save "`offdeal_clean'"

********************************************************************************
* D. Acquired patents through deal
********************************************************************************
    use `ma', clear 

    ** Drop motorola and nuance patents
    drop if inlist(deal_id, 136, 288) 

    append using "`motorola_clean'"
    append using "`nuance_clean'"

    gen acq_type = "M&A"

    replace acquired = 1 if mi(acquired) & grant_date < acquisition_date 
    assert !mi(grant_date) & !mi(acquisition_date)
    replace acquired = 0 if mi(acquired)

********************************************************************************
* E. Combine all patents
********************************************************************************
    append using "`offdeal_clean'"

    ** Bring ult parent variable 
    preserve
        use "${dta}/01 All deals - cleaned and matched.dta", clear

        keep ult_parent deal_id
        duplicates drop 
        gisid deal_id

        tempfile aux
        save "`aux'"
    restore

    merge m:1 deal_id using "`aux'",  update replace keep(1 3 4)
    assert acq_type == "Off deal" if _merge == 1
    assert acq_type == "M&A" if _merge == 3 | _merge == 4
    assert !mi(ult_parent)

    replace acquisition_year = yofd(acquisition_date)

    bys patent_id: drop if _N == 2 & mi(modnote)
    
    
    ** Save
    rename acquisition* acq*
    dropmiss, force
    gisid patent_id
    order ult_parent acq_type acq_date acq_year deal_id patent_id acquired
    sort ult_parent acq_date deal_id patent_id
    compress 

    save "${dta}/04 All patents.dta", replace





