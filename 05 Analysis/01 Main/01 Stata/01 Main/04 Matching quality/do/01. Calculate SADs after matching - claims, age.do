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
* D. Calculate means
* ==============================================================================
    
    ** Load
    u "${temp}/01. Matched patents - by lambda, 4q.dta", clear 

    ** Locals
    local config "acq_type bl_tt pre_treatment_period base_tt_threshold lambda_val"

    ** Grant quarter
    gen treatment_quarter = qofd(acq_date)
    format treatment_quarter %tq 

    ** Rename
    rename grant_date grant_date_treated
    rename num_claims num_claims_treated

    ** Get the age and claims of controls
    preserve
        u "G:\My Drive\uc3m PhD\05 Analysis\01 Main\01 Stata\01 Main\01 Data load\00 Patents\dta\01 Patent data - without citations - only matched records - no exact match on grant year.dta", clear
        append using "G:\My Drive\uc3m PhD\05 Analysis\01 Main\01 Stata\01 Main\01 Data load\00 Patents\dta\01 Patent data - without citations - only matched records - no exact match on grant year (lam 0.6 and 0.7 version).dta"

        
        rename (patent_id num_claims patent_date) (control_id num_claims_control grant_date_control) 
        keep control_id num_claims_control grant_date_control
        duplicates drop 

        gen x = date(grant_date_control, "YMD")
        format x %td
        drop grant_date_control
        rename x grant_date_control

        tempfile aux
        save "`aux'"
    restore

    tostring control_id, replace
    merge m:1 control_id using `aux', keep(3) nogen 
    destring control_id, replace 

    ** Calculate age
    foreach s in treated control {
        gen grant_quarter_`s' = qofd(grant_date_`s')
        format grant_quarter_`s' %tq
        gen age_`s' = treatment_quarter - grant_quarter_`s'
    }

    
    ** Get size of each strata 
    gcollapse (nunique) size = treated_id, by(`config' cpc treatment_quarter) merge replace 

    tempfile base
    save "`base'"
* ==============================================================================
* C. Balance for age
* ==============================================================================

    ** Calculate absolute differences per pair
    gen d_age = abs(age_treated - age_control)

    ** Collapse to get average absolute difference per config x strata 
    gcollapse d*, by(`config' cpc treatment_quarter size)

    ** Get pooled SD pre-matching and standardize differences 
    preserve
        ** Load
        import delimited using "${out}\00 Before SMDs - all, by strata.csv" ,clear

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

        tempfile befores
        save "`befores'"

        ** Filter
        keep if regexm(covariate, "age")

        ** Calculate pooled SD
        gen pooled_sd_ = sqrt((sd_treated^2 + sd_control^2)/2)
        drop sd* smd mean*
        rename pooled_sd_ pooled_sd_age

        gisid acq_type bl_tt pre_treatment_period base_tt_threshold cpc treatment_quarter, m 

        tempfile pooled_sd
        save "`pooled_sd'"

    restore

    ** Merge
    drop if size < 5
    rename cpc cpc 
    merge m:1 acq_type bl_tt pre_treatment_period base_tt_threshold cpc treatment_quarter using `pooled_sd', assert(2 3) keep(3) nogen 

    ** Standardize absolute differences
    gen s_abs_d_age = d_age / pooled_sd_age

    ** WEighted average of standardized absolute differences 
    gcollapse (mean) s_abs* [fweight = size], by(`config')

    ** Save
    compress 
    save "${out}/01 After Standardized absolute differences - age, by strata.dta", replace

    
* ==============================================================================
* D. Claims
* ==============================================================================
    u `base', clear
    
    ** Calculate absolute differences per pair
    gen d_claim = abs(num_claims_treated - num_claims_control)

    ** Collapse to get average absolute difference per config x strata 
    gcollapse d*, by(`config' cpc treatment_quarter size)

    ** Get pooled SD pre-matching and standardize differences 
    preserve

        u `befores', clear

        ** Filter
        keep if regexm(covariate, "claim")

        ** Calculate pooled SD
        gen pooled_sd_ = sqrt((sd_treated^2 + sd_control^2)/2)
        drop sd* smd mean*
        rename pooled_sd_ pooled_sd_claim

        gisid acq_type bl_tt pre_treatment_period base_tt_threshold cpc treatment_quarter, m 

        tempfile pooled_sd
        save "`pooled_sd'"

    restore

    ** Merge
    drop if size < 5
    rename cpc cpc 
    merge m:1 acq_type bl_tt pre_treatment_period base_tt_threshold cpc treatment_quarter using `pooled_sd', assert(2 3) keep(3) nogen 

    ** Standardize absolute differences
    gen s_abs_d_claim = d_claim/ pooled_sd_claim

    ** WEighted average of standardized absolute differences 
    gcollapse (mean) s_abs* [fweight = size], by(`config')

    ** Save
    compress 
    save "${out}/01 After Standardized absolute differences - claims, by strata.dta", replace

    