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
* B. Load Patent data
* ==============================================================================
    ** Load
    use "${dl}/00 Patents/dta/01 Patent data - without citations.dta", clear
    

    ** Date
    gen grant_date = date(patent_date, "YMD")
    format grant_date %td

    **Merge with Deals dataset
    gen tass = assignee0_disamb
    rename tass assignee 

    merge m:1 assignee using `try', assert(1 3) keep(3)
