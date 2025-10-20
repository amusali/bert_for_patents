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
    gl me "${est}/02 Matching estimators"
        gl aux "${me}/_aux"
        gl do  "${me}/do"
        gl dta "${me}/dta"
        gl out "${me}/out"
        gl raw "${me}/raw"
        gl temp "${me}/temp"

    gl raw_drive "G:\My Drive\PhD Data\12 Sample Final\actual results\citation"

    ** Locals
    local list_of_maximum_periods 12 16 20 40 // quarters (i.e. 3, 4, 5, 10 years)
    local seed = 1709
    local B = 500 // number of bootstrap replications

* ==============================================================================
* B. Load matched samples with records
* ==============================================================================

    ** List files in raw folder 
    local files : dir "${raw_drive}" files "Sample - *10matches.dta", respectcase
    local total_num_files : word count `files'
    di "`total_num_files' files found in ${raw_drive}"

    local i = 0
    ** Loop over the files
    foreach file of local files{

        di in yellow "Processing file: `file'"
        
        *---------------------------------------------------------*
        * B.1. Load and clean data
        *---------------------------------------------------------*
            ** Load sample
            use "${raw_drive}\\`file'", clear 
            compress

            ** Clean
            dropmiss, force 
            dropmiss, obs force
            replace lambda = round(lambda, 0.001)

            ** Destring IDs
            destring treated_id control_id, replace

            ** Gen quarter variable understood by Stata
            gen x = quarterly(quarter, "YQ")
            format x %tq
            rename quarter quarter_str
            rename x quarter
            assert !mi(quarter)
            assert quarter <= tq(2024q4) // check no future quarters

            
            ** Make acquisition date a td variable in stead of tc
            if "`: format acq_date'" == "%tc" {
                gen acq_date_td = dofc(acq_date)
                format acq_date_td %td
                drop acq_date
                rename acq_date_td acq_date
            }

            ** Gen acq quarter
            gen acq_quarter = qofd(acq_date)
            format acq_quarter %tq
            assert !mi(acq_quarter)
            assert acq_quarter <= tq(2024q4) // check no future quarters

            ** Build Deal ID for Off-Deal patents
            if regexm("`file'", "Off deal") {
                gen deal_id = strupper(ult_parent + " - " + string(acq_quarter, "%tq"))
            }
        
        *---------------------------------------------------------*
        * B.2. Filter sample
        *---------------------------------------------------------*

            ** Calculate the latest relative quarter (i.e. max is 40q == 10 years)
            gsort lambda treated_id 
            by lambda treated_id: egen latest_relative_quarter = max(rel_q)    

            ** Drop if any patent is resold after acquisition - only happens for some of M&A patents
            cap resold_date
            if _rc == 0{
                di in red "Warning: dropping observations with resold_date"
                drop if !mi(resold_date)
            }

        *---------------------------------------------------------*
        * B.3. Estimate individual treatment effects
        *---------------------------------------------------------*

            ** Estimate individual treatment effects
            gcollapse (mean) c_hat = citations_control, by(lambda_val treated_id rel_q) merge replace
            gen te = citations_treated - c_hat
            assert !mi(te)

            ** Drop control observations
            drop *control* *dist*
            duplicates drop 
            gisid lambda treated_id rel_q

            ** Save individual TE for later analysis and bootstrapping
            local file_aux = subinstr("`file'", "Sample - ", "01 Individual point estimates - ", .)
            save "${temp}/`file_aux'", replace  
            
            ** Loop over different maximum estimation periods and estimate ATEs
            foreach max_estimation_period of local list_of_maximum_periods{
                cwf default
                di in red "Estimating for maximum estimation period of `max_estimation_period' quarters (i.e. `=`max_estimation_period'/4' years)."
                
                *** Filter sample
                drop if latest_relative_quarter < `max_estimation_period'

            *---------------------------------------------------------*
            * B.3. Calculate sample sizes and ATEs
            *---------------------------------------------------------*

                ** Calculate the sample size and merge
                gcollapse (nunique) sample_size = treated_id, by(lambda) merge replace
                gcollapse (nunique) sample_size_by_ultimate_parent = treated_id, by(lambda ult_parent) merge replace
                gcollapse (nunique) sample_size_by_deal = treated_id, by(lambda deal_id) merge replace

                ** Estimate ATE for different aggregation levels
                gcollapse (mean) ate = te, by(lambda_val rel_q) merge replace 
                gcollapse (mean) ate_by_ultimate_parent = te, by(lambda_val rel_q ult_parent) merge replace 
                gcollapse (mean) ate_by_deal = te, by(lambda_val rel_q deal_id) merge replace

            *---------------------------------------------------------*
            * B.4. Save results
            *---------------------------------------------------------*

                ** Put relevant variables into frames
                frame put lambda_val rel_q ate sample_size, into(ate)
                frame put lambda_val rel_q ult_parent ate_by_ultimate_parent sample_size_by_ultimate_parent, into(ate_by_ultimate_parent)
                frame put lambda_val rel_q deal_id ate_by_deal sample_size_by_deal, into(ate_by_deal)

                ** Save ATE by lambda x relative quarter
                cwf ate

                duplicates drop 
                gisid lambda_val rel_q 

                local file_aux = subinstr("`file'", "Sample - ", "01 ATE by lambda x relative quarter - ", .)
                save "${dta}/`max_estimation_period' periods/`file_aux'", replace

                ** Save ATE by lambda x relative quarter x ultimate parent
                cwf ate_by_ultimate_parent

                duplicates drop 
                gisid lambda_val rel_q ult_parent 
                rename sample_size_by_ultimate_parent sample_size

                local file_aux = subinstr("`file'", "Sample - ", "02 ATE by lambda x relative quarter x ultimate parent - ", .)
                save "${dta}/`max_estimation_period' periods/`file_aux'", replace

                ** Save ATE by lambda x relative quarter x deal
                cwf ate_by_deal

                duplicates drop 
                gisid lambda_val rel_q  deal_id 
                rename sample_size_by_deal sample_size

                local file_aux = subinstr("`file'", "Sample - ", "03 ATE by lambda x relative quarter x deal - ", .)
                save "${dta}/`max_estimation_period' periods/`file_aux'", replace

                ** Clear 
                cwf default
                frame drop ate
                frame drop ate_by_ultimate_parent
                frame drop ate_by_deal
            }
            ** Clear memory
            clear all

            ** Update counter
            local i = `i' + 1
            di in red "`i' out of `total_num_files' files processed."            
    }
    
