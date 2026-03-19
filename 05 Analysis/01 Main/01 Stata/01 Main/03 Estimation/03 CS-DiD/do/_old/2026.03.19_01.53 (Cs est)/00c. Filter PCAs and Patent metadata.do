* ==============================================================================
* A. Set paths
* ==============================================================================
    clear all
    capture log close
    set more off
    set type double, perm
    set excelxlsxlargefile on

    ** set path
    gl analysis "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main"
    gl python "${analysis}/00 Python data"
    gl stata "${analysis}/01 Stata"
    
    gl est "${stata}/01 Main/03 Estimation"
    gl csdid "${est}/03 CS-DiD"
        gl aux "${csdid}/_aux"
        gl do  "${csdid}/do"
        gl dta "${csdid}/dta"
        gl out "${csdid}/out"
        gl raw "${csdid}/raw"
        gl temp "${csdid}/temp"
        gl log "${csdid}/log"

    gl raw_drive "G:\My Drive\PhD Data\12 Sample Final\actual results\citation_noexactmatch_on_grantyear"
    gl pca_drive "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main\00 Python data\01 CLS embeddings"

* ==============================================================================
* B. Load matched files
* ==============================================================================

    ** Load 
    u "${raw_drive}/00 Matched IDs - no pair info.dta", clear

    ** get IDs
    rename id patent_id
    keep patent_id 
    duplicates drop
    tostring patent_id, replace

    tempfile matched_ids
    save "`matched_ids'"

    ** Get grant date of controls
    mkf patents
    cwf patents 
     
    u "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main\01 Stata\01 Main\01 Data load\00 Patents\dta\01 Patent data - without citations.dta", clear

    merge 1:1 patent_id using `matched_ids', assert(1 3) keep(3) nogen

    save "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main\01 Stata\01 Main\01 Data load\00 Patents\dta\01 Patent data - without citations - only matched records - no exact match on grant year.dta", replace


    ** PCAs
    mkf pca
    cwf pca 

    import delimited using "${pca_drive}\pca_10D.csv", clear

    merge 1:1 patent_id using `matched_ids', assert(1 3) keep(3) nogen
    compress 

    save "${pca_drive}\pca_10D - only matched records - no exact match on grant year.dta", replace

