* ==============================================================================
* A. Set paths
* ==============================================================================

    clear all

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
* B. Load treated patents data incl. closest patents
* ==============================================================================
    ** Load
    import excel using "${raw}\Patents with their closest.xlsx", firstrow clear

    ** Clean
    duplicates drop
    duplicates tag patent_id, gen(dup)
    drop if dup 
    drop dup
    gisid patent_id

    ** Dates
    foreach var of varlist  *date* DateEffective{
        gen aux = dofc(`var')
        drop `var'
        rename aux `var'
        format `var' %td
    }

    ** Save
    tempfile to_merge
    save "`to_merge'"

    ** Load processed patents data
    use "${dta}\03 Treated patents - before acquisition.dta", clear

    ** Clean
    duplicates drop
    dropmiss, force

    ** Merge
    merge 1:1 patent_id using `to_merge', update assert(1 2 3 5) keep(2 3) gen(merge_closest)
    
    assert _N == 6409

    ** replace before after indicators
    replace before_acquisition = grant_date <= DateEffective if !mi(DateEffective) & mi(before_acquisition)
    replace after_acquisition = grant_date > DateEffective if !mi(grant_date) & mi(after_acquisition)


    ** Keep only before 
    keep if before_acquisition

    ** Save
    compress
    save "${dta}\04 Combined patents - without citations.dta", replace
