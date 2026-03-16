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
    use "${temp}/01. Matched patents - by lambda, 4q, all combined.dta", clear

    ** Bring metadata
    tostring patent_id, replace 
    merge m:1 patent_id using "G:\My Drive\uc3m PhD\PhD Data\09 Acquired patents\04 All patents.dta", gen(aux)
    keep if aux == 3 & treated == 1
   

    ** Get PCAs
    preserve
        /* *import delimited using "G:\My Drive\uc3m PhD\PhD Data\01 CLS Embeddings\All embeddings - float16\PCA\pca_10D.csv", clear
        u "G:\My Drive\uc3m PhD\05 Analysis\01 Main\00 Python data\01 CLS embeddings\pca_10D - only matched records - no exact match on grant year.dta", clear
        append using "G:\My Drive\uc3m PhD\05 Analysis\01 Main\00 Python data\01 CLS embeddings\pca_10D - only matched records - no exact match on grant year (lam 0.6 and 0.7 version).dta"
        append using "G:\My Drive\uc3m PhD\05 Analysis\01 Main\00 Python data\01 CLS embeddings\pca_10D - only matched records.dta" */

        import delimited using "G:\My Drive\uc3m PhD\PhD Data\01 CLS Embeddings\All embeddings - float16\PCA\pca_30D - matched records, 4q, 80.csv", clear

        isid patent_id 
        
        tempfile pcas
        save "`pcas'"
    restore

    merge m:1 patent_id using `pcas', assert(2 3) keep(3) nogen 
wdad
* ==============================================================================
* D. Calculate means
* ==============================================================================

    ** Collapse 
    collapse (mean) pc* (sd) sd_pc1=pc1 sd_pc2=pc2 sd_pc3=pc3 sd_pc4=pc4 sd_pc5=pc5 /// 
    sd_pc6=pc6 sd_pc7=pc7 sd_pc8=pc8 sd_pc9=pc9 sd_pc10=pc10, /// 
    by( acq_type bl_tt pre_treatment_period base_tt_threshold lambda_val treated)

    ** Reshape
    rename *pc* *pc*_    
    reshape wide *pc*, i(acq_type bl_tt pre_treatment_period base_tt_threshold lambda_val ) j(treated)

    ** Pooled SD
    foreach v in pc1 pc2 pc3 pc4 pc5 pc6 pc7 pc8 pc9 pc10 {
        gen pooled_`v' = sqrt((sd_`v'_1^2 + sd_`v'_0^2)/2)
        gen smd_`v'    = (`v'_1 - `v'_0) / pooled_`v'
        gen abs_smd_`v' = abs(smd_`v')
    }

    keep acq_type bl_tt pre_treatment_period base_tt_threshold lambda_val smd*
    order acq_type bl_tt pre_treatment_period base_tt_threshold lambda_val *smd*

    ** Save
    compress
    save "${out}\01 After SMDs - PCAs.dta", replace