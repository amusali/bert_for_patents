* ==============================================================================
* A. Set paths
* ==============================================================================
    clear all
    capture log close
    set more off
    set type double, perm
    set excelxlsxlargefile on

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

    gl raw_drive "G:\My Drive\PhD Data\12 Sample Final\actual results\citation_noexactmatch_on_grantyear"
    gl pca_drive "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main\00 Python data\01 CLS embeddings"

    ** Config
    local pre_treatment_periods 4 6  // 4, 6, 8, 10, 12 quarters
    local seed = 1709 // seed: periska hbd
    local acq_types = `" "Off deal" "' // Acquistion type: M&A or Off deal
    local calipers  `" "0.1000" "0.0500"  "'  //  "0.1000" "0.0500" 2.5%, 5%, 7.5%, 10%
    local base_tt = "baseline" // baseline or top-tech
    local base_tt_threshold = 80 // only used if base_tt is "top-tech"

    local lambdas 0.25 0.75 // numlist(0.0(0.05)1.0) 

    local last_post_treatment_period = 12 // last estimation period

    local pca_dimension = 10 // PCA dimensions to load

    ** Start log
    log using "${log}/99. Describe estimates.log", replace
    timer clear 1
    timer on 1

********************************************************************************
* B. Load estimates and parse results
********************************************************************************
    ** Create empty dataset to store results
    mkf est 
    mkf last_est

    cwf est 

    ** Pretrend p values
    gen pre_trend_all_pval = .
    gen pre_trend_excl_last_pval = .
    
    ** Simple ATT estimates
    gen simple = . 
    gen simple_se = . 
    gen simple_p = . 

    ** Command used to calculate CS-DiD estimates
    gen command = ""

    ** Event study estimates - // fixed for now
    forvalues t = -7 / 12 { 
            local t = string(`t')
            local t = "`=subinstr("`t'", "-", "m", .)'"

            gen es_`t' = . 
            gen es_`t'_se = . 
            gen es_`t'_p = . 
        }


    ** Get the ster files in the out folder
    local ster_files : dir "${out}\est" files "*.ster", respectcase

    ** Get the number of files
    local n_files : word count `ster_files'

    local i  1

    ** Loop over ster files
    foreach f of local ster_files {

        ** Parse file name to get parameters
        local filename = subinstr("`f'", ".ster", "", .)
        local filename = subinstr("`filename'", "05 CSDID Estimates - ", "", .)
        local filename = subinstr("`filename'", "06 CSDID Estimates - ", "", .)
        local filename = subinstr("`filename'", "06b CSDID Estimates - ", "", .)
        local filename = subinstr("`filename'", "07 CSDID Estimates - ", "", .)
        local filename = subinstr("`filename'", "- all patents, for csdid - ", ",", .)
        local filename = subinstr("`filename'", ".dta", "", .)


        qui{

            frame last_est: {
                clear
                ** Extract parameters from filename
                insobs 1 
                gen config = "`filename'"

                ** Parse
                split config, parse(",")

                ** Extract parameters
                if regexm(config[1], "baseline") {
                    drop config 
                    rename (config*) (acq_type bl_tt pretreatment_periods caliper est_config lambda posttreatment_periods)
                }
                else {
                    drop config 
                    rename (config*) (acq_type bl_tt tt_threshold pretreatment_periods caliper est_config lambda posttreatment_periods)
                }

                ** Destring 
                replace pretreatment_periods = strtrim(subinstr(pretreatment_periods, "q", "", .))       
                replace posttreatment_periods = subinstr(posttreatment_periods, pretreatment_periods[1], "", .)
                replace posttreatment_periods = subinstr(posttreatment_periods, " p - - ", "", .)

                replace caliper = subinstr(caliper, "caliper", "", .)
                replace lambda = subinstr(lambda, "lambda", "", .)

                foreach var of varlist _all {
                    destring `var', replace
                }

                local pre  = pretreatment_periods[1]
                local post = posttreatment_periods[1]

                tempfile aux
                save "`aux'"
            }

            append using `aux'
            
            ** Load the ster file
            estimates use "${out}\est\\`f'"

            ** Pretreatment tests
                ** All pretrend p-value
                csdid_estat pretrend
                replace pre_trend_all_pval =  r(pchi2) if _n == `i'

                ** Excluding last pretreatment period (i.e. t0 - 1)
                csdid_estat pretrend, window(`=-`pre' + 1' -2)
                replace pre_trend_excl_last_pval = r(pchi2) if _n == `i'

            ** Get simple ATT estimate and its SE and p-value
                csdid_estat simple
                matrix list r(table)
                    
                replace simple = r(table)[1,1] if _n == `i'
                replace simple_se = r(table)[2,1] if _n == `i'
                replace simple_p = r(table)[4,1] if _n == `i'

            ** Event study estimates
                ** Pretrend p-values
                local counter = 0
                csdid_estat event 
                matrix list r(table)
                forvalues t = `=-`pre' + 1' / `post' {
                    local t = string(`t')
                    local t = "`=subinstr("`t'", "-", "m", .)'"
                    
                    ** Get estimates and return matrices

                    replace es_`t' = r(table)[1, 3 + `counter'] if _n == `i'
                    replace es_`t'_se = r(table)[2, 3 + `counter']  if _n == `i'
                    replace es_`t'_p = r(table)[4, 3 + `counter'] if _n == `i'

                    local counter = `counter' + 1

                }

            ** Command used
            ereturn list 
            replace command = "`e(cmdline)'" if _n == `i'
        
            ** Export Plots of event study estimates 
            csdid_estat event 
        
            csdid_plot, ytitle("ATT (in log-citations)")
            dropmiss, obs force

            local plot_filename = "`=subinstr("`filename'", "no exact matching on grant year,", "", .)'"
            graph export "${out}\graphs\ES Plot - CSDID - `plot_filename'.png", replace width(3000) height(2000)
            }
            ** Report progress and update counter
            local i = `i' + 1
            di in red "Processed file `i' of `n_files'"

        }
        


********************************************************************************
* C. Clean results
********************************************************************************    

    order est_config acq_type bl_tt tt_threshold pretreatment_periods posttreatment_periods caliper lambda command pre_trend* simple*
    sort est_config acq_type bl_tt tt_threshold pretreatment_periods posttreatment_periods caliper lambda

    ** Save final results
    save "${dta}\CSDID Estimates Summary.dta", replace