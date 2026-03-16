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
    log using "${log}\01. Calculate SMDs after matching - pretreatment citations.log", replace
    timer clear 1
    timer on 1

* ==============================================================================
* B. Reconcile before SMDs
* ==============================================================================
    ** Load
    import delimited using "${out}/00 Before SMDs - all, by strata.csv", clear
    
    ** Config
    gen acq_type = "M&A" if regexm(config, "M&A")
    replace acq_type = "Off deal" if mi(acq_type)

    gen bl_tt = "baseline" if regexm(config, "bl")
    replace bl_tt = "top-tech" if mi(bl_tt)

    gen pre_treatment_period = 4 if regexm(config, "4q")
    replace pre_treatment_period = 6 if regexm(config, "6q")
    replace pre_treatment_period = 8 if regexm(config, "8q")
    assert !mi(pre_treatment_period)

    gen base_tt_threshold = 80 if bl_tt == "top-tech"

    ** Treatment quarter - cohort
    gen treatment_quarter = quarterly(treated_q, "YQ")
    format treatment_quarter %tq

    ** Weighted mean
    local config "acq_type bl_tt pre_treatment_period base_tt_threshold"
    gisid acq_type bl_tt pre_treatment_period base_tt_threshold cpc treatment_quarter covariate, m 
    gcollapse (mean) smd* [fweight = n_treated], by(`config' covariate)

    gen lambda_val = 99 // code for before matching

    ** Rename for merging later
    replace cov = "age" if cov == "age_q"
    replace cov = "num_claim" if cov == "num_claims"
    replace cov = subinstr(cov, "PC", "pc", .)
    keep if pre_treatment_period == 4
    
    tempfile before
    save "`before'"

* ==============================================================================
* C. Reconcile after SMDs
* ==============================================================================
    ** Load
    u "${out}/99. SMDs after - PCs.dta", clear

    merge 1:1 acq_type bl_tt pre_treatment_period base_tt_threshold lambda_val using "${out}/99. SMDs after - claims, age.dta", assert(3) nogen
    merge 1:1 acq_type bl_tt pre_treatment_period base_tt_threshold lambda_val using "${out}/99. SMDs after - citations.dta", assert(3) nogen

    ** Reshape
    reshape long smd_ , i(acq_type bl_tt pre_treatment_period base_tt_threshold lambda_val) j(covariate) str

    ** Add before 
    rename smd_ smd 
    append using `before'
    gsort acq_type bl_tt pre_treatment_period base_tt_threshold  covariate lambda_val

    save "${out}/02. SMDs.dta", replace
    awd