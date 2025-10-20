* ==============================================================================
* A. Set paths
* ==============================================================================
    clear all
    capture log close
    set more off
    set type double, perm
    set excelxlsxlargefile on
    set maxvar 32767, perm
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

    cd "${out}\est"

    gl raw_drive "G:\My Drive\PhD Data\12 Sample Final\actual results\citation"
    gl pca_drive "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main\00 Python data\01 CLS embeddings"

    ** Config
    local pre_treatment_periods 4 6 8 10 // 4, 6, 8, 10, 12 quarters
    local seed = 1709 // seed: periska hbd
    local acq_types = `" "Off deal" "M&A" "' // Acquistion type: M&A or Off deal
    local calipers  `" "0.0750" "0.1000" "0.0500"  "'  //  "0.1000" "0.0500" 2.5%, 5%, 7.5%, 10%
    local base_tt = "top-tech" // baseline or top-tech
    local base_tt_threshold = 90 // only used if base_tt is "top-tech"

    local lambdas 0 0.5 1 // numlist(0.0(0.05)1.0) 


    local pca_dimension = 10 // PCA dimensions to load

    ** Start log
    log using "${log}/04. Estimate - only age and active covariates.log", replace
    timer clear 1
    timer on 1

* ==============================================================================
* B. Load PCAs of patents with 10 dimensions and CPCs & merge
* ==============================================================================
    /* *---------------------------------------------------------*
    * B.1. Load PCA data with 10 dimensions and merge
    *---------------------------------------------------------*
        ** Load
        mkf pca
        cwf pca

        use "${pca_drive}\pca_`pca_dimension'D - only matched records.dta", clear

        tempfile pca
        save "`pca'"

       
    *---------------------------------------------------------*
    * B.2. Load CPC data and merge
    *---------------------------------------------------------*
        * Get grant date of controls
        mkf patents
        cwf patents 
        
        u "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main\01 Stata\01 Main\01 Data load\00 Patents\dta\01 Patent data - without citations - only matched records.dta", clear

        ** Filter and make Grant date a Stata quarter date
        keep patent_id patent_date cpc_subclass_current
        duplicates drop 
        gisid patent_id

        gen grant_date = date(patent_date, "YMD")
        format grant_date %td
        drop patent_date

        gen grant_quarter = qofd(grant_date)
        format grant_quarter %tq

        gen grant_year = yofd(grant_date)

        tempfile patents
        save "`patents'"
        
        ** Merge back into main data
        cwf default 
        */
* ==============================================================================
* B. Load CS-DiD compliant data per config
* ==============================================================================

    foreach pre_treatment_period of local pre_treatment_periods { 
        foreach acq_type of local acq_types {
            foreach caliper of local calipers {
                foreach lambda of local lambdas {

                    ** Set the last post-treatment period based on acq type
                    if "`acq_type'" == "M&A" {
                        local last_post_treatment_period = 11
                    }
                    else if "`acq_type'" == "Off deal" {
                        local last_post_treatment_period = 12
                    }
                    else {
                        di in red "Error: acq_type must be either 'M&A' or 'Off deal'"
                        exit
                    }

                    ** Get the filename - e.g. "01 Sample - M&A, baseline, 4q, caliper0.1000 - all patents, for csdid.dta" or "01 Sample - Off deal, top-tech, 80, 4q, caliper0.0250 - all patents, for csdid.dta"
                    if "`base_tt'" == "baseline" {
                        ** Adjust filename 
                        local filename = "01 Sample - `acq_type', `base_tt', `=string(`pre_treatment_period', "%2.0f")'q, caliper`caliper' - all patents, for csdid.dta"
                    }
                    else if "`base_tt'" == "top-tech" {
                        ** Adjust filename 
                        local filename = "01 Sample - `acq_type', `base_tt', `=string(`base_tt_threshold', "%2.0f")', `=string(`pre_treatment_period', "%2.0f")'q, caliper`caliper' - all patents, for csdid.dta"
                    }
                    else {
                        di in red "Error: base_tt must be either 'baseline' or 'top-tech'"
                        exit
                    }
                
                    di in red "`filename'"

                * ==============================================================================

                    ** Skip if already estimated
                    local est_range_str = " -`pre_treatment_period' - `last_post_treatment_period'" // range for plots
                    local est_filename = subinstr("`filename'", "01 Sample", "04 CSDID Estimates", .)
                    local est_filename = subinstr("`est_filename'", "", "", .)
                    local est_filename = subinstr("`est_filename'", " - all patents, for csdid.dta", "", .)

                    di "Will save into ${out}\est\\`est_filename', lambda`lambda', p`est_range_str'.ster"
                    capture confirm file "${out}\est\\`est_filename', lambda`lambda', p`est_range_str'.ster"
                    if _rc == 0 {
                        di "Estimation `f' exists — skipping."
                        continue   // skip this iteration
                    }

                    ** Conditional load per lambda
                    cwf default 
                    use if lambda == `lambda' using "${dta}/`filename'", clear

                    ** Drop varaibles that are present only for acquired patents
                    drop config ult_parent deal_id num_claims assignee cpc_*

                    ** Gen logs
                    gen log_citation = log(1 + citation)

                * ==============================================================================
                * C. Filter sample
                * ==============================================================================

                    ** Age
                    gen age = quarter - qofd(grant_date)
                    gen age_sq = age^2

                    ** Filter on age 
                    assert age >= 0 & !mi(age)
                    *keep if inrange(age, 0, 80) // patents have a life of 20 years = 80 quarters
                    gen active = age <= 80

                    ** Drop treated patents for which we cannot observe the full post-treatment period 
                    drop if treated == 1 & cohort + `last_post_treatment_period' > tq(2024q4)

                    ** Relative quarter 
                    gen quarter_to_treatment = quarter - qofd(acq_date)
                    keep if inrange(quarter_to_treatment, -`pre_treatment_period', `last_post_treatment_period') | mi(acq_date) // drop periods before pre-treatment periods
                    drop acq_date

                    ** Drop cohorts that do not have many patents to avoid noisy estimates
                    local num_periods_per_treated = `pre_treatment_period' + `last_post_treatment_period' + 1 // max number of periods per treated patent in the data (used in csdid)
                    bys cohort: gen num_patents_in_cohort = _N

                    *** Check that when `num_patents_in_cohort' is a multiple of `num_periods_per_treated'
                    assert mod(num_patents_in_cohort, `num_periods_per_treated') == 0 if treated == 1
                    drop if num_patents_in_cohort < 10 * `num_periods_per_treated' & treated == 1 // at least 10 patents per cohort

                    ** Full sample - wont´do anything but kept for clarity
                    drop if grant_quarter + `last_post_treatment_period' > tq(2024q4) // drop patents that cannot be observed until the end of the last post-treatment period

                    ** Drop the control x quarter cells in the beginning and ends of the timeline for which the csdid package cannot construct valid estimates
                    count if treated
                    gcollapse (sum) num_treated = treated, by(quarter) merge replace
                    drop if num_treated == 0
                    count if treated

                    
                * ==============================================================================
                * D. Estimate CS-DiD
                * ==============================================================================

                    ** Sanity checks
                    gisid  patent_id quarter
                    assert !mi(patent_id) & !mi(quarter) & !mi(citation) & !mi(cohort)
                    assert citation >= 0

                    ** Estimate
                    di in red "Estimating CS-DiD with `pre_treatment_period' pre-treatment periods"

                    ** Run csdid
                    csdid log_citation age* active, i(patent_id) g(cohort) t(quarter) method(reg) seed(`seed') 
                    
                    ** Save estimates
                    estimates save "${out}/est//`est_filename', lambda`lambda', p`est_range_str'", replace

                }
            } 
        }
    }

    timer off 1
    timer list

    log close