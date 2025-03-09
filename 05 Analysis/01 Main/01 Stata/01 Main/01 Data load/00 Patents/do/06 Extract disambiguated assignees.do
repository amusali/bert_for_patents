* ==============================================================================
* A. Set paths
* ==============================================================================

    ** set path
    gl analysis "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main"
    gl python "${analysis}/00 Python data"
    gl stata "${analysis}/01 Stata"
    
    gl dl "${stata}/01 Main/01 Data load/00 Patents"
        gl do "${dl}/do"
        gl dta "${dl}/dta"
        gl out "${dl}/out"
        gl raw "${dl}/raw"

* ==============================================================================
* B. Load Patent data
* ==============================================================================
    ** Load dta
    use "${dta}/01 Patent data - without citations.dta", clear

    ** Keep assignees
    keep assignee*_disamb
    duplicates drop

    ** Reshape
    gen version = _n
    rename *_disamb disamb_*

    reshape long disamb_assignee, i(version) j(k)
    drop if mi(dis)
    drop v k
    duplicates drop 
    sort d 
    
    ** save
    compress
    export delimited using "${out}/USPTO disambiguated assignees.csv", replace

********************************************************************************
* C. Extract patent count 
********************************************************************************