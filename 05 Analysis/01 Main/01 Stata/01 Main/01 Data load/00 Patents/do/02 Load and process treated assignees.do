
* ==============================================================================
* A. Set paths
* ==============================================================================

    ** Set paths
    gl analysis "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main"
    gl python "${analysis}/00 Python data"
    gl stata "${analysis}/01 Stata"
    
    gl dl "${stata}/01 Main/01 Data load"
        gl do "${dl}/do"
        gl dta "${dl}/dta"
        gl out "${dl}/out"
        gl raw "${dl}/raw"

* ==============================================================================
* B. Load treated assignees data
* ==============================================================================

    ** Load
    import excel using "${raw}\Deals with treated assignees.xlsx", firstrow  clear

    ** Keep relevant vars
    keep AcquirorFullName AcquirorUltimateParent TargetFullName TargetUltimateParent DateAnnounced DateEffective Assignees



    tempfile aux
    save "`aux'"

    ** Load another file - possibly duplicated rows
    import excel using "${raw}\Deals with treated assignees - 2.xlsx", firstrow clear

    ** Keep relevant vars
    keep AcquirorFullName AcquirorUltimateParent TargetFullName TargetUltimateParent DateAnnounced DateEffective Assignees* DealSynopsis
    duplicates drop

    ** Motorola
    replace Assignees_x = Assignees_y if Assignees_y == "Motorola Mobility LLC"

    ** Clean
    drop if mi(Assignees_x)
    drop Assignees_y
    duplicates drop 
    rename Assignees_x Assignees

    ** Combine
    append using "`aux'"

    ** Drop irrelevant deals
    drop if TargetFullName == "Apple Computer Inc"
    drop if TargetUltimateParent == "HTC Corp"

    
    ** Duplicates at levels except DealSynopsis
    qui ds DealSynopsis, not
    bys `r(varlist)' (DealSynopsis): keep if _n == _N
    duplicates drop 

    ** Date vars
    foreach var in DateAnnounced DateEffective{
        gen aux = dofc(`var')
        drop `var'
        rename aux `var'
        format %td `var'
    }

    ** Save 
    compress
    save "${dta}/02 Deals with treated assignees.dta", replace