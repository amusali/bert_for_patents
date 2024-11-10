
* ==============================================================================
* A. Set paths
* ==============================================================================

    ** set path
    gl main "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main"
    gl python "${main}/00 Python data"
    gl stata "${main}/01 Stata"

    gl raw "${stata}/raw/sdc_gafam"

* ==============================================================================
* B. Load data
* ==============================================================================

    clear all
    mkf existing
    cwf existing

    ** load already checked SDC deals
    import excel "${python}/sdc_google_combined.xlsx", firstrow clear

    mkf alldeals
    cwf alldeals

    ** load all deals
    import excel "${raw}/SDC Platinum sdc_hightech_filter.xlsx", sheet("Request 7") cellrange(A3:AP50465) firstrow clear

********************************************************************************
* C. Clean existing deals
********************************************************************************
    cwf existing

    ** Buybacks
    count if AcquirorFullName == TargetFullName
    assert `r(N)' == 1 // only one buyback deal

    drop if AcquirorFullName == TargetFullName

    ** Duplicates 
    duplicates drop 

    ** Get gafam CUSIP
    preserve
        keep Acquiror6digitCUSIP
        duplicates drop 

        tempfile gafam
        save "`gafam'"
    restore


********************************************************************************
* D. Clean all deals and combine with existing
********************************************************************************
    cwf alldeals 

    ** drop duplicates
    duplicates drop

    ** filter by GAFAm
    merge m:1 Acquiror6digitCUSIP using "`gafam'", keep(match)

    ** buybacks
    drop if AcquirorFullName == TargetFullName

********************************************************************************
* E. Save
********************************************************************************
    compress
    export excel using "${python}/05 Deals/SDC gafam - all.xlsx",  firstrow(variables) replace






    
