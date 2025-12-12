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

    gl raw_drive "G:\My Drive\PhD Data\12 Sample Final\actual results\citation_noexactmatch_on_grantyear"

    ** Locals
    local list_of_maximum_periods 12 16 20 40 // quarters (i.e. 3, 4, 5, 10 years)
    local seed = 1709
    local B = 100 // number of bootstrap replications

    ** Start log
    log using "${log}/00b. Add citations and consolidate.log", replace

* ==============================================================================
* B. Load citations first
* ==============================================================================

    ** Load 
    mkf citations
    cwf citations 

    import delimited using "${raw_drive}/collapsed_citations.csv", clear // it is the only version despite its name

    ** Quarter variable
    gen quarter = quarterly(citation_quarter, "YQ")
    format quarter %tq
    drop citation_quarter

    rename citation_count citation

    ** Checks and save
    assert !mi(patent_id) & !mi(quarter) & !mi(citation)
    assert citation > 0
    gisid patent_id quarter

    tempfile citations
    save "`citations'"

********************************************************************************
* C. Load matched sample with all patents and save per each config
********************************************************************************

    mkf matched
    cwf matched

    ** Load
    u "${dta}/00 Sample - without citations - no exact matching on grant year.dta", clear

    ** ID check
    gisid config lambda patent_id

    ** Cohort variable
    gen cohort = qofd(acq_date)
    replace cohort = 0 if mi(cohort)
    format cohort %tq
    label var cohort "Cohort for csdid (grant quarter)"

    ** Get levels of config 
    qui levelsof config, local(configs) 

    local total_num_configs : word count `configs'
    di "`total_num_configs' configurations found"

    local counter = 0
    ** Loop over the configurations
    foreach config of local configs {

        di in red "Processing config: `config'"

        ** Keep only the config and put into a new frame
        frame put if config == "`config'", into(sample)

        frame sample:{
            
            ** ID check
            gisid lambda patent_id

            ** Make patent ID numeric
            destring patent_id, replace
            
            ** Expand until 2024Q4
            gen expand = tq(2024q4) - grant_quarter + 1
            expand expand

            gen quarter = grant_quarter
            format quarter %tq
            bys lambda patent_id : replace quarter = quarter[_n-1] + 1 if _n > 1
            drop expand

            ** Merge citations for treated patents
            merge m:1 patent_id quarter using "`citations'", assert(1 2 3) keep(1 3) nogen
            replace citation = 0 if mi(citation)

            ** Save sample
            compress
            gisid lambda patent_id quarter

            assert !mi(citation) & !mi(cohort) & !mi(grant_quarter) & !mi(patent_id) & !mi(lambda) & !mi(quarter) & !mi(treated) 
            assert citation >= 0
            assert grant_quarter <= quarter & quarter <= tq(2024q4)

            local config = subinstr("`config'", ", 10matches.pkl", "", .)
            local config = subinstr("`config'", "_", "", .)

            save "${dta}/01 Sample - `config' - all patents, for csdid - no exact matching on grant year.dta", replace
        }

        ** Drop the frame
        frame drop sample

        ** Progress
        local counter = `counter' + 1
        di in red "`counter' out of `total_num_configs' configurations processed."

    }

    ** Finish log
    log close

