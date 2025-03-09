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
* B. Load & harmonize all deals data
* ==============================================================================
    
    *---------------------------------------------------------*
    * B.1. GAFAM Empire
    *---------------------------------------------------------*
        use "${dta}/01 GAFAM empire - processed.dta", clear

        ** Source
        gen source = "GAFAM Empire"

        tempfile gafamempire
        save "`gafamempire'"

    *---------------------------------------------------------*
    * B.2. DISCERN
    *---------------------------------------------------------*
        use "${dta}/DISCERN - deals and assignees.dta", clear

        ** Source 
        gen source = "DISCERN"

        rename target parent 
        drop acq_sub*

        tempfile discern
        save "`discern'"

    *---------------------------------------------------------*
    * B.3. SDC
    *---------------------------------------------------------*
        use "${dta}/03 SDC data.dta", clear

        ** Source 
        gen source = "SDC Platinum" 

        ** Dates
        replace DateEffective = dofc(DateEffective)
        replace DateAnnounced = dofc(DateAnnounced)

        ** rename
        rename ( AcquirorUltimateParent AcquirorFullName TargetFullName DateEffective DealSynopsis DealValueUSDMillions) ///
        (ult_parent parent child  acquisition_date  deal_synopsis deal_value)

        tempfile sdc
        save "`sdc'"

    *---------------------------------------------------------*
    * B.4. G&L
    *---------------------------------------------------------*
        use "${dta}/04 G&L deals.dta", clear

        ** Source 
        gen source = "G&L (2017)" 

        ** Clean
        drop Fundinground
        rename (Totalfunding acquisition_date) (deal_value acquisition_year)
        replace deal_value = deal_value / 10^6

        tempfile gl
        save "`gl'"

    *---------------------------------------------------------*
    * B.5 Aguirre
    *---------------------------------------------------------*
        use "${dta}/05 Aguirre deals.dta", clear

        ** Source 
        gen source = "Aguirre (2024)" 

        tempfile aguirre
        save "`aguirre'"

********************************************************************************
* C. Combine
********************************************************************************
    use `gafamempire', clear
    append using `discern'
    append using `sdc'
    append using `gl'
    append using `aguirre'

    assert !mi(source)

    ** Light cleaning
    replace child = strupper(child)
    qui ds *, has(type string)
    foreach var in `r(varlist)'{
        replace `var' = strtrim(`var')
    }

    order source deal_id ult_parent parent child acquisition_date acquisition_year deal_synopsis

    ** Save
    compress
    save "${out}/06 All deals.dta", replace

********************************************************************************
* D. Extract target names
********************************************************************************
    keep child
    duplicates drop
    export delimited using "${out}/ All deals target names.csv", replace
