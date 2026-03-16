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
    log using "${log}\01. Calculate SMDs after matching - PCAs.log", replace
    timer clear 1
    timer on 1

* ==============================================================================
* B. Load files for all lambdas and combine
* ==============================================================================
    use "${temp}/01. Matched patents - by lambda, 4q.dta", clear

    ** Locals
    local config "acq_type bl_tt pre_treatment_period base_tt_threshold lambda_val"

    ** Grant quarter
    gen treatment_quarter = qofd(acq_date)
    format treatment_quarter %tq 

    ** PCAs of treateds
    rename treated_id patent_id

    ** Get PCAs
    preserve
        /* *import delimited using "G:\My Drive\uc3m PhD\PhD Data\01 CLS Embeddings\All embeddings - float16\PCA\pca_10D.csv", clear
        u "G:\My Drive\uc3m PhD\05 Analysis\01 Main\00 Python data\01 CLS embeddings\pca_10D - only matched records - no exact match on grant year.dta", clear
        append using "G:\My Drive\uc3m PhD\05 Analysis\01 Main\00 Python data\01 CLS embeddings\pca_10D - only matched records - no exact match on grant year (lam 0.6 and 0.7 version).dta"
        append using "G:\My Drive\uc3m PhD\05 Analysis\01 Main\00 Python data\01 CLS embeddings\pca_10D - only matched records.dta" */

        import delimited using "G:\My Drive\uc3m PhD\PhD Data\01 CLS Embeddings\All embeddings - float16\PCA\pca_30D - matched records, 4q, 80.csv", clear
        
        tempfile pcas
        save "`pcas'"
    restore

    merge m:1 patent_id using `pcas', assert(2 3) keep(3) nogen 

    ** Rename
    rename pc# pc#_treated 
    rename patent_id treated_id
    rename control_id patent_id

    ** PCAs of controls
    merge m:1 patent_id using `pcas', assert(2 3) keep(3) nogen 
    
    ** Rename    
    rename patent_id control_id
    rename pc# pc#_control

    ** Get size of each strata 
    gcollapse (nunique) size = treated_id, by(`config' cpc treatment_quarter) merge replace 

    tempfile base
    save "`base'"

* ==============================================================================
* D. Calculate means
* ==============================================================================

    
    ** Calculate absolute differences per pair
    forvalues i = 1/30{
        gen d_pc`i' = abs(pc`i'_treated - pc`i'_control)
    }

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
        keep if regexm(covariate, "PC")

        ** Calculate pooled SD
        gen pooled_sd_ = sqrt((sd_treated^2 + sd_control^2)/2)
        drop sd* smd mean*
        
        ** Reshape
        reshape wide pooled_sd_, i(acq_type bl_tt pre_treatment_period base_tt_threshold cpc treatment_quarter) j(covariate) str

        gisid acq_type bl_tt pre_treatment_period base_tt_threshold cpc treatment_quarter, m 

        tempfile pooled_sd
        save "`pooled_sd'"

    restore

    ** Merge
    drop if size < 5
    rename cpc cpc 
    merge m:1 acq_type bl_tt pre_treatment_period base_tt_threshold cpc treatment_quarter using `pooled_sd', assert(2 3) keep(3) nogen 

    ** Standardize absolute differences
    forvalues i = 1/30{
        gen s_abs_d_pc`i' = d_pc`i' / pooled_sd_PC`i'
    }

    ** WEighted average of standardized absolute differences 
    gcollapse (mean) s_abs* [fweight = size], by(`config')

    ** Save
    compress 
    save "${out}/01 After Standardized absolute differences - PCs, by strata.dta", replace

    