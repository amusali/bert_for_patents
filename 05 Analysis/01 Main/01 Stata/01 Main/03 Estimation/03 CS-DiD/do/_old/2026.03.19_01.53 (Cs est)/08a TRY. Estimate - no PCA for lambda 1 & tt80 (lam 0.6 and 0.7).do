* ==============================================================================
* A. Set paths
* ==============================================================================
    clear all
    capture log close
    set more off
    set type double, perm
    set excelxlsxlargefile on

    ** set path
    gl analysis "G:\My Drive\uc3m PhD\05 Analysis\01 Main"
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

    gl raw_drive "G:\My Drive\uc3m PhD\PhD Data\12 Sample Final\actual results\citation_noexactmatch_on_grantyear"
    gl pca_drive "G:\My Drive\uc3m PhD\05 Analysis\01 Main\00 Python data\01 CLS embeddings"

    ** Config
    local pre_treatment_periods 4 // 4, 6, 8, 10, 12 quarters
    local seed = 1709 // seed: periska hbd
    local acq_types = `" "M&A" "' // Acquistion type: M&A or Off deal
    local calipers  `"  "0.0500"  "'  //  "0.1000" "0.0500" 2.5%, 5%, 7.5%, 10%
    local base_tt = "top-tech" // baseline or top-tech
    local base_tt_threshold = 80 // only used if base_tt is "top-tech"

    local lambdas 0.6  // numlist(0.0(0.05)1.0) 

    local last_post_treatment_period = 11 // last estimation period

    local pca_dimension = 10 // PCA dimensions to load

    ** Start log
    log using "${log}/08a TRY. Estimate - no PCA for lambda 1 & tt80 (MA lam 0.6 and 0.7) dropped cohorts.log", replace
    timer clear 1
    timer on 1

* ==============================================================================
* B. Load CPCs & merge
* ==============================================================================

    *---------------------------------------------------------*
    * B.1. Load PCA data with 10 dimensions and merge
    *---------------------------------------------------------*
        ** Load

        use "${pca_drive}\pca_`pca_dimension'D - only matched records - no exact match on grant year (lam 0.6 and 0.7 version).dta", clear

        tempfile pca
        save "`pca'"

       
    *---------------------------------------------------------*
    * B.2. Load CPC data and merge
    *---------------------------------------------------------*
        * Get grant date of controls

        u "G:\My Drive\uc3m PhD\05 Analysis\01 Main\01 Stata\01 Main\01 Data load\00 Patents\dta\01 Patent data - without citations - only matched records - no exact match on grant year (lam 0.6 and 0.7 version).dta", clear

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
       
* ==============================================================================
* B. Load CS-DiD compliant data per config
* ==============================================================================

    foreach pre_treatment_period of local pre_treatment_periods { 
        foreach acq_type of local acq_types {
            foreach caliper of local calipers {
                foreach lambda of local lambdas {

                    ** Get the filename - e.g. "01 Sample - M&A, baseline, 4q, caliper0.1000 - all patents, for csdid.dta" or "01 Sample - Off deal, top-tech, 80, 4q, caliper0.0250 - all patents, for csdid.dta"
                    if "`base_tt'" == "baseline" {
                        ** Adjust filename 
                        local filename = "01 Sample - `acq_type', `base_tt', `=string(`pre_treatment_period', "%2.0f")'q, caliper`caliper' - all patents, for csdid - nemogy - add lambdas.dta"
                    }
                    else if "`base_tt'" == "top-tech" {
                        ** Adjust filename 
                        local filename = "01 Sample - `acq_type', `base_tt', `=string(`base_tt_threshold', "%2.0f")', `=string(`pre_treatment_period', "%2.0f")'q, caliper`caliper' - all patents, for csdid - nemogy - add lambdas.dta"
                    }
                    else {
                        di in red "Error: base_tt must be either 'baseline' or 'top-tech'"
                        exit
                    }
                
                    di in red "`filename'"



                    ** Skip if already estimated
                    local est_range_str = " -`pre_treatment_period' - `last_post_treatment_period'" // range for plots
                    local est_filename = subinstr("`filename'", "01 Sample", "08a TRY CSDID Estimates", .)
                    local est_filename = subinstr("`est_filename'", "", "", .)
                    local est_filename = subinstr("`est_filename'", " - all patents, for csdid - nemogy - add lambdas.dta", "", .)

                    di "Will save into ${out}\est\\`est_filename', lambda`lambda', p`est_range_str'.ster"
                    capture confirm file "${out}\est\\`est_filename', lambda`lambda', p`est_range_str'.ster"
                    if _rc == 0 {
                        di "Estimation `f' exists — skipping."
                        continue   // skip this iteration
                    }


                    ** Conditional load per lambda
                    use if lambda == `lambda' using "${dta}/`filename'", clear

                    ** Merge PCA and CPC data
                    tostring patent_id, replace
                    merge m:1 patent_id using "`pca'", assert(2 3) keep(3) nogen 

                    merge m:1 patent_id using "`patents'", update replace assert(2 3 4) keep(3 4) nogen 
                    destring patent_id, replace

                    ** CPC
                    encode cpc_subclass_current, gen(cpc)

                    ** Drop varaibles that are present only for acquired patents
                    drop config ult_parent deal_id num_claims assignee cpc_*

                    ** Gen logs
                    gen log_citation = log(1 + citation)

                * ==============================================================================
                * C. Filter sample and create covariates
                * ==============================================================================
                    ** Age
                    gen age = quarter - qofd(grant_date)
                    gen age_sq = age^2

                    ** Create active variable based on age 
                    assert age >= 0
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

                    drop if num_patents_in_cohort < 20 * `num_periods_per_treated' & treated == 1 // at least 10 patents per cohort

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
                    assert citation >= 0 & log_citation >= 0

                    ** Estimate
                    di in red "Estimating CS-DiD with `pre_treatment_period' pre-treatment periods"

                    ** Run csdid
                    drop if inlist(cohort, 211, 238)

                    if `lambda' == 1 {
                        di in red "Estimating CS-DiD without PCA controls -> lambda 1"
                        csdid log_citation age* i.grant_year i.cpc active, i(patent_id) g(cohort) t(quarter) method(reg) seed(`seed') 
                    }
                    else {
                        di in red "Estimating CS-DiD with PCA controls"
                        csdid log_citation  pc* i.grant_year ,  g(cohort) t(quarter) method(reg) seed(`seed') 
                    }
                    wr3wr123daw
                    ** Save
                    estimates save "${out}/est//`est_filename', lambda`lambda', p`est_range_str'_dropped cohorts", replace

                    ** Check pretrend
                    csdid_estat pretrend
                    csdid_estat event 
                    csdid_plot, title("Event study - `est_filename', lambda `lambda'") saving("${out}/graphs//`est_filename', lambda`lambda', p`est_range_str' - event plot.gph", replace)
                    awdaw
                }
            } 
        }
    }

    timer off 1
    timer list

    log close

/*
    Event Study:Dynamic effects
------------------------------------------------------------------------------
             |      Coef.   Std. Err.      z    P>|z|     [95% Conf. Interval]
-------------+----------------------------------------------------------------
     Pre_avg |  -.0159543   .0068418    -2.33   0.020    -.0293641   -.0025446
    Post_avg |  -.1313866    .015196    -8.65   0.000    -.1611703   -.1016029
         Tm3 |   -.014843   .0206988    -0.72   0.473    -.0554119    .0257259
         Tm2 |  -.0061618    .020928    -0.29   0.768      -.04718    .0348564
         Tm1 |  -.0268583   .0206397    -1.30   0.193    -.0673113    .0135948
         Tp0 |  -.0898586   .0214345    -4.19   0.000    -.1318696   -.0478477
         Tp1 |  -.0700206   .0220257    -3.18   0.001    -.1131901   -.0268511
         Tp2 |  -.0966844   .0215197    -4.49   0.000    -.1388623   -.0545065
         Tp3 |  -.0804458   .0222105    -3.62   0.000    -.1239776   -.0369141
         Tp4 |  -.1331242   .0201186    -6.62   0.000     -.172556   -.0936924
         Tp5 |  -.1312692   .0210478    -6.24   0.000    -.1725221   -.0900163
         Tp6 |  -.1348581   .0211373    -6.38   0.000    -.1762864   -.0934298
         Tp7 |  -.1308754   .0205387    -6.37   0.000    -.1711305   -.0906203
         Tp8 |  -.1738313   .0203102    -8.56   0.000    -.2136386   -.1340239
         Tp9 |   -.159666   .0207805    -7.68   0.000     -.200395    -.118937
        Tp10 |  -.1783748   .0203257    -8.78   0.000    -.2182123   -.1385372
        Tp11 |  -.1976311   .0193198   -10.23   0.000    -.2354972   -.1597651
------------------------------------------------------------------------------

. csdid_estat attgt
------------------------------------------------------------------------------
             |      Coef.   Std. Err.      z    P>|z|     [95% Conf. Interval]
-------------+----------------------------------------------------------------
g203         |


 Dynamic Effects:

┌────────────┬──────────┬────────────┬────────────────────────────┐
│ Event time │ Estimate │ Std. Error │ [95% Pointwise Conf. Band] │
├────────────┼──────────┼────────────┼────────────────────────────┤
│         -3 │  -0.0214 │     0.0204 │ [-0.0791,  0.0363]         │
│         -2 │  -0.0088 │     0.0207 │ [-0.0674,  0.0499]         │
│         -1 │  -0.0325 │     0.0205 │ [-0.0904,  0.0254]         │
│          0 │  -0.0862 │     0.0212 │ [-0.1461, -0.0262] *       │
│          1 │  -0.0656 │     0.0217 │ [-0.1270, -0.0041] *       │
│          2 │  -0.0974 │     0.0213 │ [-0.1577, -0.0371] *       │
│          3 │  -0.0809 │     0.0220 │ [-0.1430, -0.0187] *       │
│          4 │  -0.1327 │     0.0199 │ [-0.1889, -0.0765] *       │
│          5 │  -0.1270 │     0.0208 │ [-0.1858, -0.0682] *       │
│          6 │  -0.1400 │     0.0209 │ [-0.1992, -0.0809] *       │
│          7 │  -0.1324 │     0.0202 │ [-0.1895, -0.0752] *       │
│          8 │  -0.1793 │     0.0201 │ [-0.2362, -0.1225] *       │
│          9 │  -0.1640 │     0.0205 │ [-0.2220, -0.1060] *       │
│         10 │  -0.1799 │     0.0200 │ [-0.2364, -0.1235] *       │
│         11 │  -0.1612 │     0.0212 │ [-0.2212, -0.1012] *       │
└────────────┴──────────┴────────────┴────────────────────────────┘