
* ==============================================================================
* A. Set paths
* ==============================================================================

    clear all
    capture log close
    set more off
    set type double, perm
    set excelxlsxlargefile on

 
    gl google_drive "G:\My Drive\uc3m PhD"
    gl analysis "${google_drive}\05 Analysis\01 Main"
    gl python "${analysis}\00 Python data"
    gl stata "${analysis}\01 Stata"
    
    gl dl "${stata}/01 Main/01 Data load"
    gl desc "${dl}/03 Descriptives"
        gl aux "${desc}/_aux"
        gl do  "${desc}/do"
        *gl dta "${desc}/dta"
        gl out "${desc}/out"
        gl raw "${desc}/raw"
        gl temp "${desc}/temp"
    gl dta "${stata}/01 Main/03 Estimation/03 CS-DiD/dta"



********************************************************************************
* C. Descriptives on samples
********************************************************************************
    
    *---------------------------------------------------------*
    * C.1. M&A
    *---------------------------------------------------------*
    ** Load data with all samples
    use "${dta}\02 All samples - acquired patents with citations.dta", clear

    ** Filter patent with sufficent post-horizon (the last available quarter is 2024Q3 so the last acquisition should be in 2022Q1)
    keep if qofd(acq_date) <= 248 

    ** Filter M&A
    keep if sample == "M&A - baseline"
    assert !mi(sample)
    
    ** Filter
    keeporder ult_parent deal_id patent_id 
    duplicates drop
    gisid patent_id

    ** Calculate number of patents acquired per deal and sort
    gcollapse (nunique) count = patent, by(ult deal)
    gsort  -count

    ** Calculate shares
    egen total = sum(count)
    gen share = count / total
    gen cum_total = share[1] if _n == 1
    replace cum_total = share[_n] + cum_total[_n-1] if _n >1 

    keep if _n <= 10 // top 10 deals

    merge 1:m deal_id using "G:\My Drive\uc3m PhD\05 Analysis\01 Main\01 Stata\01 Main\01 Data load\02 Combine\dta\01 All deals - cleaned and matched.dta", assert( 2 3) keep(3)

    ** Keep relevant tables manually
    sort deal_id
    keep if inlist(_n, 2, 3, 8, 11, 14, 19, 20, 21, 26, 53)
    assert _N == 10
    gisid deal_id
    gsort -count


    ** Homogenize
    replace child = "Motorola Mobility" if _n == 1
    replace child = "Nuance Communications" if _n == 2
    replace child = "Fitbit" if _n == 3


    ** Filter to top 5
    keep if _n <= 5
    drop parent
    dropmiss, force
    
    save "${out}\03 Top 5 M&A deals by patent portfolio.dta", replace

    *---------------------------------------------------------*
    * C.2. Off deal 
    *---------------------------------------------------------*

     ** Load data with all samples
    use "${dta}\02 All samples - acquired patents with citations.dta", clear

    ** Filter patent with sufficent post-horizon (the last available quarter is 2024Q3 so the last acquisition should be in 2022Q1)
    keep if qofd(acq_date) <= 248 

    ** Filter M&A
    keep if sample == "Off deal - baseline"
    assert !mi(sample)

    ** Assign deal ID manually for off deal transfers
    replace deal_id = 991 if ult_parent == "GOOG" 

    replace deal_id = 992 if ult_parent == "APPL" & acq_date == td(11may2012)
    replace deal_id = 993 if ult_parent == "APPL" & acq_date == td(11feb2013)
    replace deal_id = 994 if ult_parent == "APPL" & yofd(acq_date) >= 2019

    replace deal_id = 995 if ult_parent == "FB" & acq_date == td(27mar2012)
    replace deal_id = 996 if ult_parent == "FB" & acq_date >= td(4jun2012)

    replace deal_id = 997 if ult_parent == "MSFT" & inlist(acq_date, td(30sep2011), td(30mar2012), td(26jun2012))
    replace deal_id = 998 if ult_parent == "MSFT" & inlist(acq_date, td(10may2012), td(19jun2012))
    replace deal_id = 999 if ult_parent == "MSFT" & inlist(acq_date, td(14jun2012), td(15jun2012))

    assert !mi(deal_id) 

    ** Add the seller info
    gen assignor = ""
    replace assignor = "IBM" if inlist(deal_id, 991, 995, 997)
    replace assignor = "ROCKSTAR BIDCO" if inlist(deal_id, 992, 998)
    replace assignor = "AOL" if inlist(deal_id, 996, 999)
    replace assignor = "EASTMAN KODAK COMPANY" if deal_id == 993
    replace assignor = "INTEL CORPORATION" if deal_id == 994
    assert !mi(assignor)

    
    ** Filter
    keeporder ult_parent deal_id patent_id assignor
    duplicates drop
    gisid patent_id

    ** Calculate number of patents acquired per deal and sort
    gcollapse (nunique) count = patent, by(ult deal assignor)
    gsort  -count

    ** Calculate shares
    egen total = sum(count)
    gen share = count / total
    gen cum_total = share[1] if _n == 1
    replace cum_total = share[_n] + cum_total[_n-1] if _n >1 

    save "${out}\03 Top Off deal transfers by patent portfolio.dta", replace


