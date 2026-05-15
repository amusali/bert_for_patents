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
        gl out "${csdid}/out_colab"
        gl raw "${csdid}/raw"
        gl temp "${csdid}/temp"
        gl log "${csdid}/log"


    cd "${out}\est\nearby patents"

    ** Start log
    log using "${log}/99 COLAB. Describe estimates (nearby patents).log", replace
    timer clear 1
    timer on 1

********************************************************************************
* B. Load estimates and parse results
********************************************************************************

    ** Get the ster files in the out folder
    local csv_es_files : dir "${out}\est\nearby patents" files "*event study.csv", respectcase

    ** Get the number of files
    local n_files : word count `csv_es_files'
    di in red "There are " + `n_files'

    local i  1

    ** Loop over ster files
    foreach f of local csv_es_files {
        qui{
        clear

          
            preserve
                clear
                ** Extract parameters from filename
                insobs 1 
                gen config = "`f'"

                ** Parseawdw
                split config, parse(",")

                ** Extract parameters
                if regexm(config[1], "baseline") {
                    drop config 
                    rename (config*) (acq_type bl_tt pretreatment_periods caliper est_config lambda posttreatment_periods outcome_var)
                }
                else {
                    drop config 
                    rename (config*) (acq_type bl_tt tt_threshold pretreatment_periods caliper est_config lambda posttreatment_periods outcome_var)
                }
                ** Destring 
                replace pretreatment_periods = strtrim(subinstr(pretreatment_periods, "q", "", .))       
                replace posttreatment_periods = subinstr(posttreatment_periods, pretreatment_periods[1], "", .)
                replace outcome_var = subinstr(outcome_var, "- event study.csv", "", .)

                replace posttreatment_periods = subinstr(posttreatment_periods, " p- - ", "", .)

                replace caliper = subinstr(caliper, "caliper", "", .)
                replace lambda = subinstr(lambda, "lambda", "", .)

                foreach var of varlist _all {
                    destring `var', replace
                }

                tostring lam, replace

                local pre  = pretreatment_periods[1]
                local post = posttreatment_periods[1]

                tempfile aux
                save "`aux'"
            restore

            append using `aux'
            
            preserve
                
                ** Load the ster file
                import delimited using  "${out}\est\nearby patents\\`f'", clear
                sort event_time

                ** Get simple ATT estimate and its SE and p-value
                gcollapse (mean) simple = att, by(treatment_status) merge replace
                replace simple = simple[_N]

                ** Event study estimates
                tostring event, replace 
                replace event = subinstr(event, "-", "m", .)
                drop treatment_status
                gen id = 1
                reshape wide att se ci_lower ci_upper p_value wald_stat simple, i(id) j(event) str 
                drop id 
                
                rename att* es_*
                rename se* es_*_se
                rename ci_lower* es_*_cl
                rename ci_upper* es_*_cu

                rename wald_stat0 pre_trend_walt_statistic
                drop wald_stat*

                rename p_value0 pre_trend_all_pval
                drop p_value*

                rename simple0 est_simple
                drop simple*
                
                order est_simple pre* es*

                tempfile estimates
                save "`estimates'"

            restore
            ** Report progress and update counter

            merge 1:1 _n using `estimates', assert(3) nogen 



            tempfile filename`i'
            save "`filename`i''"
            di in red "Processed file `i' of `n_files'"

            local i = `i' + 1

        }
    }

    

     ** Loop over ster files
    forvalue i = 1/`n_files' {
            append using  `filename`i''
        }

    gen contains_pcs = lambda != "1"
    replace lam = subinstr(lam, " with PCs", "", .)
    destring lam, replace 
    replace est_config = strtrim(est_config)

    duplicates drop 

********************************************************************************
* C. Clean results
********************************************************************************    

    order est_config acq_type bl_tt tt_threshold pretreatment_periods posttreatment_periods caliper lambda contains_pcs  pre_trend* *simple*
    sort est_config acq_type bl_tt tt_threshold pretreatment_periods posttreatment_periods caliper lambda contains_pcs

    ** Save final results
    save "${dta}\CSDID Estimates Summary - Nearby Patents - (COLAB, Cross Section estimates).dta", replace
    awd