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

    ** Filter
    drop if cosine_distance < 0.01 
    drop if mahalanobis_distance > 10
    
    ** Locals
    local config "acq_type bl_tt pre_treatment_period base_tt_threshold lambda_val"

    ** Grant quarter
    gen treatment_quarter = qofd(acq_date)
    format treatment_quarter %tq 

    ** Get citations for treated and control separately 
    foreach s in treated control{
        replace `s'_vector = subinstr(`s'_vector, "[", "", .)
        replace `s'_vector = subinstr(`s'_vector, "]", "", .)
        split `s'_vector, parse(".")
        drop `s'_vector
        rename `s'_vector* (`s'_cit_m4 `s'_cit_m3 `s'_cit_m2 `s'_cit_m1)
        destring `s'_cit*, replace
    }

    ** Get size of each strata 
    gcollapse (nunique) size = treated_id, by(`config' cpc treatment_quarter) merge replace 

    ** Calculate the weight of each treated id
    gen weight = 1
    gcollapse (sum) weight , by(`config' treated_id) merge replace
    replace weight = 1 / weight

    *** TRY
    gcollapse *_cit_m*, by(`config' cpc treatment_quarter size)

    
    forvalues i = 1/4{
        gen smd_cit_m`i' = treated_cit_m`i' - control_cit_m`i'
    }

    
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
        keep if regexm(covariate, "cit_m")
        drop if inlist(cov, "cit_m5", "cit_m6", "cit_m7", "cit_m8")

        ** Calculate pooled SD
        gen pooled_sd_ = sqrt((sd_treated^2 + sd_control^2)/2)
        drop sd* smd mean*

        ** Reshape
        reshape wide pooled_sd_, i( acq_type bl_tt pre_treatment_period base_tt_threshold cpc treatment_quarter) j(covar) str

        gisid acq_type bl_tt pre_treatment_period base_tt_threshold cpc treatment_quarter, m 

        tempfile pooled_sd
        save "`pooled_sd'"

    restore

    ** Merge
    drop if size < 5
    rename cpc cpc 
    merge m:1 acq_type bl_tt pre_treatment_period base_tt_threshold cpc treatment_quarter using `pooled_sd', assert(2 3) keep(3) nogen 


    ** Add SDs
    forvalues i = 1/4{
        replace smd_cit_m`i'  = smd_cit_m`i'  / pooled_sd_cit_m`i'
    }

    gcollapse (mean) smd* [fweight = size], by(`config')

    save "${out}/99. SMDs after - citations.dta", replace 
    

* ==============================================================================
* C. PCs
* ==============================================================================
    ** Load
    u "${temp}/01. Matched patents - by lambda, 4q.dta", clear 

    ** Filter
    drop if cosine_distance < 0.01
    drop if mahalanobis_distance > 10

    ** Locals
    local config "acq_type bl_tt pre_treatment_period base_tt_threshold lambda_val"

    ** Grant quarter
    gen treatment_quarter = qofd(acq_date)
    format treatment_quarter %tq 

    ** PCAs of treateds
    rename treated_id patent_id

    ** Get PCAs
    preserve

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

    *** TRY
    gcollapse pc*, by(`config' cpc treatment_quarter size)

    
    forvalues i = 1/30{
        gen smd_pc`i' = pc`i'_treated - pc`i'_control
    }

    ** Get pooled SD pre-matching and standardize differences 
    preserve
        ** Load
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


        ** Filter
        keep if regexm(covariate, "PC")

        ** Calculate pooled SD
        gen pooled_sd_ = sqrt((sd_treated^2 + sd_control^2)/2)
        drop sd* smd mean*

        ** Reshape
        reshape wide pooled_sd_, i( acq_type bl_tt pre_treatment_period base_tt_threshold cpc treatment_quarter) j(covar) str

        gisid acq_type bl_tt pre_treatment_period base_tt_threshold cpc treatment_quarter, m 

        tempfile pooled_sd
        save "`pooled_sd'"

    restore

    ** Merge
    drop if size < 5
    rename cpc cpc 
    merge m:1 acq_type bl_tt pre_treatment_period base_tt_threshold cpc treatment_quarter using `pooled_sd', assert(2 3) keep(3) nogen 

    ** Add SDs
    forvalues i = 1/30{
        replace smd_pc`i'  = smd_pc`i'  / pooled_sd_PC`i'
    }

    gcollapse (mean) smd* [fweight = size], by(`config')

    save "${out}/99. SMDs after - PCs.dta", replace
 

* ==============================================================================
* D. Claims
* ==============================================================================

     ** Load
    u "${temp}/01. Matched patents - by lambda, 4q.dta", clear 

    ** Filter
    drop if cosine_distance < 0.01
    drop if mahalanobis_distance > 10

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

    *** TRY
    gcollapse num_claim* age* , by(`config' cpc treatment_quarter size)

    gen smd_age = age_treated - age_control
    gen smd_num_claim = num_claims_treated - num_claims_control

    ** Get pooled SD pre-matching and standardize differences 
    preserve
        ** Load
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


        ** Filter
        keep if regexm(covariate, "PC") | regexm(covariate, "claim") | regexm(covariate, "age")

        ** Calculate pooled SD
        gen pooled_sd_ = sqrt((sd_treated^2 + sd_control^2)/2)
        drop sd* smd mean*

        ** Reshape
        reshape wide pooled_sd_, i( acq_type bl_tt pre_treatment_period base_tt_threshold cpc treatment_quarter) j(covar) str

        gisid acq_type bl_tt pre_treatment_period base_tt_threshold cpc treatment_quarter, m 

        tempfile pooled_sd
        save "`pooled_sd'"

    restore

    ** Merge
    drop if size < 5
    rename cpc cpc 
    merge m:1 acq_type bl_tt pre_treatment_period base_tt_threshold cpc treatment_quarter using `pooled_sd', assert(2 3) keep(3) nogen 

    ** Add SDs
    replace smd_age = smd_age / pooled_sd_age
    replace smd_num_claim = smd_num_claim / pooled_sd_num_claim

    gcollapse (mean) smd* [fweight = size], by(`config')

    save "${out}/99. SMDs after - claims, age.dta", replace
    exit


