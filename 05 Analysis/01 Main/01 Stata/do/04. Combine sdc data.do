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

    foreach var in DateEffective DateAnnounced DealValueasofDate{
        replace `var' = dofc(`var')
    }

    tempfile try
    save "`try'"

    ** load all deals
    mkf alldeals
    cwf alldeals
    import excel "${python}/05 Deals/SDC gafam - all.xlsx", firstrow clear

* ==============================================================================
* C. Generate variables
* ==============================================================================

    ** Clean alldeals
    cwf alldeals
    bys Acquiror6digitCUSIP AcquirorFullName Target6digitCUSIP TargetFullName (Date*): keep if _n == _N
     
    drop _merge
    merge 1:m  Acquiror6digitCUSIP AcquirorFullName Target6digitCUSIP TargetFullName using `try'