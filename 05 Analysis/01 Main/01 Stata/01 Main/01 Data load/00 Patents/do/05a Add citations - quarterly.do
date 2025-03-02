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
* B. Load combined patents
* ==============================================================================
    ** Load
    use "${dta}\04 Combined patents - without citations.dta", clear

    *---------------------------------------------------------*
    * B.1. Add citations for treated patents
    *---------------------------------------------------------*

    merge 1:m patent_id using "${dta}\02 Patent citations - raw.dta"
