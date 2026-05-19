* ==============================================================================
* A. Set paths
* ==============================================================================
    clear all
    capture log close
    set more off
    set type double, perm
    set excelxlsxlargefile on

    ** set path
   
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

    ** Locals
    local list_of_maximum_periods 12 16 20 40 // quarters (i.e. 3, 4, 5, 10 years)
    local seed = 1709
    local B = 100 // number of bootstrap replications

* ==============================================================================
* B. Load all the IDs from matched records
* ==============================================================================
 
    ** Load data for lambda 0 and 1
    use if inlist(lam, 0, 1) using "${raw_drive}/00 Matched IDs - no pair info.dta", clear

    ** Bring filename
    merge m:1 file_id using "${raw_drive}/00 Matched IDs - file map.dta", assert(3) nogen

    ** Load data for lamda 0.6 and 0.7 and clean
    preserve
        use "G:\My Drive\uc3m PhD\PhD Data\12 Sample Final\actual results\paper\00 Matched IDs - no pair info.dta", clear
        merge m:1 file_id using "G:\My Drive\uc3m PhD\PhD Data\12 Sample Final\actual results\paper\00 Matched IDs - file map.dta", assert(3) nogen

        tempfile intermediate_lamdas
        save "`intermediate_lamdas'"
    restore

    append using `intermediate_lamdas'

    gisid file_id lambda_val id 
    drop path
    rename filename config

    destring id, replace
    compress

    keep if treated == 0
    drop treated

* ==============================================================================
* C. Loop over configs x treated flag x lambdas to calculate overlaps
* ==============================================================================

    ** Locals
    keep if regexm(config, "4q") & regexm(config, "caliper_0.05")
    drop if regexm(config, "M&A") & lambda == 0.7      
    drop if regexm(config, "Off deal") & lambda == 0.6

    tempfile main
    save "`main'"

    levelsof file_id, local(files)
    levelsof config, local(configs)
    levelsof lambda_val, local(lambdas)
    foreach cfg of local configs{
        preserve
            ** Get the filename
            keep if config == "`cfg'"
            local filename = config[1]
            local filename = subinstr("`filename'", ".pkl", ".csv", .)
            local file = file_id[1]
        restore

        if regexm("`filename'", "lambda"){
            preserve
                ** Load 
                import delimited using "G:\My Drive\uc3m PhD\PhD Data\11 Matches\actual results\paper\\`filename'", clear
                
                ** ID check
                gisid treated_id control_id lam

                ** Calculate weight 
                bys treated_id lam: gen t = _N
                gen weight = 1 / t
                gcollapse (sum) weight, by(control_id lam)
                gen file_id  = `file'

                tempfile file_id_`file'_A
                save "`file_id_`file'_A'"
            restore
        }
        else{
            preserve
                ** Load 
                import delimited using "G:\My Drive\uc3m PhD\PhD Data\11 Matches\actual results\citation_no_exact_match_on_grantyear\\`filename'", clear

                ** ID check
                gisid treated_id control_id lam

                ** Calculate weight 
                bys treated_id lam: gen t = _N
                gen weight = 1 / t
                gcollapse (sum) weight, by(control_id lam)
                gen file_id  = `file'

                tempfile file_id_`file'_B
                save "`file_id_`file'_B'"
            restore
        }
        

    }
    clear
    foreach file of local files{

            append using `file_id_`file'_A'
            append using `file_id_`file'_B'

        
    }
    
    keep if inlist(lam, 0, 0.6, 0.7, 1)
    /* clear
    foreach file of local files{
            append using `file_id_`file'' 
            } */
        

    rename control_id id
    merge 1:1 file_id lambda_val id using `main', assert(1 3) keep(3) nogen
    
    ** Reshape
    tostring lam, replace
    replace lam = subinstr(lam, ".", "", .)
    replace lam = "inter" if inlist(lam, "6", "7")

    drop config 

    reshape wide weight, i(file id) j(lam) str

    ** Calculate effective number of contrrols
    foreach setting in "1" "0" "inter"{
        preserve
            keep id file weight`setting'
            drop if mi(weight)
            gisid id file
            
            gcollapse (sum) sum_of_weights = weight, by(file) merge replace 
            gen nominator = sum_of_weights ^ 2

            gen weight_sq = weight ^ 2
            gcollapse (sum) denominator = weight_sq, by(file) merge replace 

            gen effective_controls = nominator / denominator
            bys file: gen sample_count = _N

            gen share = effective_controls  /  sample_count 

            keeporder file effective_controls sample_count share 

            gen setting = "`setting'"

            duplicates drop 

            tempfile set_`setting'
            save "`set_`setting''" 

        restore
    }

    clear
    foreach setting in "1" "0" "inter"{
        append using `set_`setting''
    }

    sort file_id setting 

    merge m:1 file_id using "${raw_drive}/00 Matched IDs - file map.dta", assert(2 3) keep(3) nogen

    awdwa
