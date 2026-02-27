* ==============================================================================
* A. Set paths
* ==============================================================================
    clear all
    capture log close
    set more off
    set type double, perm
    set excelxlsxlargefile on

    ** Set path
    gl google_drive "G:\My Drive\uc3m PhD"
    gl analysis "${google_drive}\05 Analysis\01 Main"
    gl python "${analysis}\00 Python data"
    gl stata "${analysis}\01 Stata"
    
    gl est "${stata}\01 Main"
    gl csdid "${est}\04 Matching quality"
        gl aux "${csdid}\_aux"
        gl do  "${csdid}\do"
        gl dta "${csdid}\dta"
        gl out "${csdid}\out"
        gl raw "${csdid}\raw"
        gl temp "${csdid}\temp"
        gl log "${csdid}\log"

    gl raw_drive "${google_drive}\PhD Data\12 Sample Final\actual results\citation_noexactmatch_on_grantyear"
    gl pca_drive "${google_drive}\05 Analysis\01 Main\00 Python data\01 CLS embeddings"

    gl matches "${google_drive}\PhD Data\11 Matches\actual results"

    ** Config
    local pre_treatment_periods 4  // 4, 6, 8, 10, 12 quarters
    local seed = 1709 // seed: periska hbd
    local acq_types = `" "M&A" "Off deal" "' // Acquistion type: M&A or Off deal
    local calipers  `" "0.0500"  "'  //  "0.1000" "0.0500" 2.5%, 5%, 7.5%, 10%
    local base_tt = `"  "baseline" "top-tech" "' // baseline or top-tech
    local base_tt_threshold = 80 // only used if base_tt is "top-tech"

    local lambdas 0 0.6 0.7 1 // numlist(0.0(0.05)1.0) 

    local last_post_treatment_period = 12 // last estimation period

    local pca_dimension = 10 // PCA dimensions to load

    ** Start log
    log using "${log}\02. Compare SMDs for citations.log", replace
    timer clear 1
    timer on 1

* ==============================================================================
* B. Load SMDs brfore
* ==============================================================================
    ** Load and get config from source file
    import excel using "${out}/00 Before SMDs - unweighted.xlsx", firstrow clear

    ** Config 
    replace source = subinstr(source, "precomputed_mahalanobis_", "", .)
    replace source = subinstr(source, ".pkl", "", .)

    gen acq_type = "M&A" if regexm(source, "M&A")
    replace acq_type = "Off deal" if mi(acq_type)

    gen bl_tt = "baseline" if regexm(source, "_bl_")
    replace bl_tt = "top-tech" if mi(bl_tt)

    gen base_tt_threshold = 80 if bl_tt == "top-tech"

    gen pre_treatment_period = 4 if regexm(source, "4q")
    replace pre_treatment_period = 6 if regexm(source, "6q")
    replace pre_treatment_period = 8 if regexm(source, "8q")
    assert !mi(pre_treatment_period)

    ** Rename
    rename (SMD abs_SMD) (smd_ abs_smd_)
    keep acq_type bl_tt base_tt_threshold pre_treatment_period smd* quarter

    ** Reshape
    replace quarter = subinstr(quarter, "t-", "m", .)
    reshape wide *smd*, i(acq_type bl_tt base_tt_threshold pre_treatment_period) j(quarter) str

    gen config = acq + " | " + bl + " | " + "before"

* ==============================================================================
* C. Merge with after SMDs of pre treatment citations
* ==============================================================================

    keep if pre == 4
    append using "${out}\01 After SMDs - citations.dta"
    
    replace config = acq + " | " + bl + " | " + string(lambda) if mi(config)
    reshape long smd_, i(config) j(cov) str

    save "${dta}/01. Pre treatment citations - SMDs comparison.dta", replace
    
