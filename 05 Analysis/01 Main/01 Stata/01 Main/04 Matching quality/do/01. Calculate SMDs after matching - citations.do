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
                        keep treated* control* lambda
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
                        keep treated* control* lambda
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

    preserve
        keep treated* lambda acq_type bl_tt pre_treatment_period base_tt_threshold 

        rename (treated_id treated_vector) (patent_id citations)

        gen treated = 1

        tempfile treated
        save "`treated'"
    restore
    drop treated* 
    rename (control_id control_vector) (patent_id citations)
    gen treated = 0
    append using `treated'

    ** Save
    save "${temp}/01. Matched patents - by lambda, 4q, all combined.dta", replace
awd
    // This is the file that feeds into "01. Calculate SMDs after matching - PCs.do" but it has not been run yet.
* ==============================================================================
* D. Calculate means
* ==============================================================================
    ** get citations
    replace citations = subinstr(citations, "[", "", .)
    replace citations = subinstr(citations, "]", "", .)
    split citations, parse(".")
    drop citations
    rename citations* (cit_m4 cit_m3 cit_m2 cit_m1)
    destring cit*, replace
    ** Collapse 
    collapse (mean) cit* (sd) sd_cit_m4 = cit_m4 sd_cit_m3 = cit_m3 sd_cit_m2 = cit_m2 sd_cit_m1 = cit_m1, by( acq_type bl_tt pre_treatment_period base_tt_threshold lambda_val treated)

    ** Reshape
    reshape wide *cit*, i(acq_type bl_tt pre_treatment_period base_tt_threshold lambda_val ) j(treated)

    ** Pooled SD
    foreach v in cit_m1 cit_m2 cit_m3 cit_m4 {
        gen pooled_`v' = sqrt((sd_`v'1^2 + sd_`v'0^2)/2)
        gen smd_`v'    = (`v'1 - `v'0) / pooled_`v'
        gen abs_smd_`v' = abs(smd_`v')
    }

    keep acq_type bl_tt pre_treatment_period base_tt_threshold lambda_val *smd*
    order acq_type bl_tt pre_treatment_period base_tt_threshold lambda_val *smd*

    rename *smd_cit_* *smd_*

    ** Save
    compress
    save "${out}\01 After SMDs - citations.dta", replace


/*


tempfile base
save `base'
levelsof lamb, local(lvals)
levelsof acq, local(acqs)
levelsof bl, local(bls)
foreach acq of local acqs{
foreach bl of local bls{
 
 di in red "Acq type: `acq'; bl_tt: `bl'"
foreach l1 of local lvals {
    foreach l2 of local lvals {
    preserve
        keep if acq_type = "`acq'" & bl_tt == "`bl'"
        use `base', clear
        
        gen in_l1 = (lambda_val == `l1')
        gen in_l2 = (lambda_val == `l2')
        
        collapse (max) in_l1 in_l2, by(control_id)
        
        gen both = in_l1 & in_l2
        
        quietly summarize both
        di "Overlap between lambda `l1' and `l2' = " r(sum)
        restore

}
}
}
}