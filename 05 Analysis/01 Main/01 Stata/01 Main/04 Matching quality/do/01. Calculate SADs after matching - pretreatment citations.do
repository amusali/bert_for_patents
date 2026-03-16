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
* B. Load files for all lambdas and combine
* ==============================================================================

    foreach pre_treatment_period of local pre_treatment_periods { 
        foreach acq_type of local acq_types {
            foreach caliper of local calipers {
                foreach base of local base_tt{
                    preserve
                    *---------------------------------------------------------*
                    * B.1. Locate the filename and load
                    *---------------------------------------------------------*
                        ** Get the filename - e.g. "01 Sample - M&A, baseline, 4q, caliper0.1000 - all patents, for csdid.dta" or "01 Sample - Off deal, top-tech, 80, 4q, caliper0.0250 - all patents, for csdid.dta"
                        if "`base'" == "baseline" {
                            ** Adjust filename 
                            local filename = "01 Hybrid matches - `acq_type', `base', `=string(`pre_treatment_period', "%2.0f")'q, caliper_`caliper', 10matches.csv"
                        }
                        else if "`base'" == "top-tech" {
                            ** Adjust filename 
                            local filename = "01 Hybrid matches - `acq_type', `base', `=string(`base_tt_threshold', "%2.0f")', `=string(`pre_treatment_period', "%2.0f")'q, caliper_`caliper', 10matches.csv"
                        }
                        else {
                            di in red "Error: base_tt must be either 'baseline' or 'top-tech'"
                            exit
                        }

                        di in red "`filename'"

                        ** Load
                        import delimited using "${matches}\citation_no_exact_match_on_grantyear\\`filename'", clear

                        ** Filter lambdas
                        keep if inlist(lambda, 0, 1)
                        keep treated* control* lambda maha* cosine*
                        duplicates drop 

                        ** Config
                        gen acq_type = "`acq_type'"
                        gen bl_tt = "`base'"
                        gen pre_treatment_period = `pre_treatment_period'
                        gen base_tt_threshold = `base_tt_threshold'

                        ** Save
                        tempfile lambdas0and1
                        save "`lambdas0and1'"

                    *---------------------------------------------------------*
                    * B.2. Locate and load 0.6 and 0.7 lambdas
                    *---------------------------------------------------------*

                        if "`base'" == "baseline" {
                            ** Adjust filename 
                            local filename = "01 Hybrid matches (lambda 0.6 and 0.7) - `acq_type', `base', `=string(`pre_treatment_period', "%2.0f")'q, caliper_`caliper', 10matches.csv"
                        }
                        else if "`base'" == "top-tech" {
                            ** Adjust filename 
                            local filename = "01 Hybrid matches (lambda 0.6 and 0.7) - `acq_type', `base', `=string(`base_tt_threshold', "%2.0f")', `=string(`pre_treatment_period', "%2.0f")'q, caliper_`caliper', 10matches.csv"
                        }
                        else {
                            di in red "Error: base_tt must be either 'baseline' or 'top-tech'"
                            exit
                        }

                        di in red "`filename'"

                        ** Load
                        import delimited using "${matches}\paper\\`filename'", clear

                        ** Checks
                        assert inlist(lam, 0.6, 0.7)
                        keep treated* control* lambda maha* cosine*
                        duplicates drop 

                        ** Config
                        gen acq_type = "`acq_type'"
                        gen bl_tt = "`base'"
                        gen pre_treatment_period = `pre_treatment_period'
                        gen base_tt_threshold = `base_tt_threshold'

                        ** Combine
                        append using `lambdas0and1'

                        tempfile aux
                        save "`aux'"
                        
                    restore

                    append using `aux'

                   

                    }

                }
            }
        }

* ==============================================================================
* C. Separate treated and controls
* ==============================================================================

    ** Fix threshold
    replace base_tt_threshold = . if bl_tt == "baseline"

    ** Bring metadata
    rename   treated_id patent_id

    tostring patent_id, replace 
    merge m:1 patent_id using "G:\My Drive\uc3m PhD\PhD Data\09 Acquired patents\04 All patents.dta", assert(2 3) keep(3) nogen
    rename  patent_id treated_id
    destring treated_id, replace

    ** Save
    keeporder acq_type bl_tt pre_treatment_period base_tt_threshold lambda_val treated_id treated_vector control_id control_vector ult_parent acq_date deal_id cpc_subclass_current maha* cosine* grant_date num_claims
    compress 
    gisid acq_type bl_tt pre_treatment_period base_tt_threshold lambda_val treated_id control_id, m
    save "${temp}/01. Matched patents - by lambda, 4q.dta", replace

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


    ** Calculate absolute differences per pair
    forvalues i = 1/4{
        gen d_m`i' = abs(treated_cit_m`i' - control_cit_m`i')
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
awd
    ** Standardize absolute differences
    forvalues i = 1/4{
        gen s_abs_d_m`i' = d_m`i' / pooled_sd_cit_m`i'
    }

    ** WEighted average of standardized absolute differences 
    gcollapse (mean) s_abs* [fweight = size], by(`config')

    ** Save
    sort acq_type bl_tt pre_treatment_period base_tt_threshold lambda_val
    compress
    save "${out}\01 After Standardized absolute differences - citations, by strata.dta", replace



