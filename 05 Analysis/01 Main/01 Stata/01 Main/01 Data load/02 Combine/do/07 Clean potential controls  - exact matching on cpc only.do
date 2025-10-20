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
* B. Load potential controls data
* ==============================================================================
    ** Load
    import delimited using "G:\My Drive\PhD Data\10 Sample - pre final\potential_controls_no_exact_match_grantyear_cpc.csv", clear

    ** Clean
    drop if mi(grant_year)
    tostring patent_id, replace

    gisid patent_id
    assert application_year <= grant_year

    assert _N == 5960763 - 19649 //coming from Colab notebook

    keep patent_id 

    tempfile potential_controls
    save "`potential_controls'"

********************************************************************************
* C. Load patents data and identify GAFAM patents 
********************************************************************************
    ** Load
    use "${dl}\00 Patents\dta\01 Patent data - without citations.dta", clear

    keep patent_id assignee0_disamb
    preserve
        keep assignee0_disamb
        duplicates drop 

        replace ass = strupper(ass)

        gen keep = regexm(ass, "GOOGLE")
        replace keep = regexm(ass, "FACEBOOK") | regexm(ass, "META PLATFORM") if keep == 0
        replace keep = regexm(ass, "AMAZON TECH") | regexm(ass, "AMAZON.COM") if keep == 0
        replace keep = regexm(ass, "MICROSOFT ") if keep == 0

        replace keep = 1 if ass == "APPLE COMPUTER INC." | ass == "APPLE INC."

        keep if keep
        drop keep 

        gen aux = ass
        drop ass

        tempfile gafam
        save "`gafam'"

    restore

    replace ass = strupper(ass)
    gen aux = ass 
    drop ass
    merge m:1 aux using `gafam', assert(1 3) keep(3)

    keep patent_id 
    gisid patent_id

    tempfile gafam_patents
    save "`gafam_patents'"

********************************************************************************
* D. Drop GAFAM patents from the list of potential controls
********************************************************************************
    use `potential_controls'

    merge 1:1 patent_id using `gafam_patents', keep(1) nogen 
    gisid patent_id 

    compress
    export delimited using "G:\My Drive/PhD Data/10 Sample - pre final/clean_potential_control_ids_no_exact_match_grantyear.csv", replace