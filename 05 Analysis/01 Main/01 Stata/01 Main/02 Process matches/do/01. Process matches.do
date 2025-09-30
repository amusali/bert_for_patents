*··············································································*

	clear all
	capture log close
	set more off 
	set type double, perm
	set excelxlsxlargefile on
    set trace off

*··············································································*
* A. Paths and other auxiliary parameters
*··············································································*

	** Global folders 
	gl raw "G:\My Drive\PhD Data\11 Matches\actual results\citation"				               
	gl out "G:\My Drive\PhD Data\12 Sample Final\actual results\citation"

********************************************************************************
* B. Load
********************************************************************************
    *---------------------------------------------------------*
    * B.1. Load the citations
    *---------------------------------------------------------*
        ** Load
        import delimited using "G:\My Drive\PhD Data\12 Sample Final\actual results\citation\collapsed_citations_old.csv", clear

        assert !mi(patent_id)
        assert !mi(citation_quarter)
        assert !mi(citation_count)

        ** Quarter
        gen quarter = quarterly(citation_quarter, "YQ")
        assert !mi(quarter)
        format quarter %tq
        rename citation_quarter quarter_str

        ** Save in two formats
        preserve
            rename (patent_id citation_count) (treated_id treated_count)

            tempfile treated
            save "`treated'"
        restore

        rename (patent_id citation_count) (control_id control_count)

        tempfile control
        save "`control'"

    *---------------------------------------------------------*
    * B.2. Load patents data
    *---------------------------------------------------------*
        ** Load
        use "G:\My Drive\PhD Data\09 Acquired patents\04 All patents.dta", clear

        ** Rename
        rename patent_id treated_id

        *check this line!!!! - assert cpc_sequence_current == 0
        keep ult_parent acq_type acq_date deal_id treated_id resold_date //acquired assignee grant_date modnote assignor resold_date cpc_group*
        
        ** clean and save
        dropmiss, force 
        dropmiss, obs force
            
        gisid treated_id
        tempfile patents
        save "`patents'"

    *---------------------------------------------------------*
    * B.2. Load the matches
    *---------------------------------------------------------*
        ** Load all the files in a folder
        local files : dir "${raw}" files "01 Hybrid matches -*10matches.csv" , respectcase
        
        local counter = 0
        mkf clean
        foreach file of local files {
            di "Processing file: `file'"
            import delimited using "${raw}\\`file'", clear 

            ** Parse sample version from the filename
            local sample_version = subinstr("`file'", "01 Hybrid matches - ", "", .)
            local sample_version = subinstr("`sample_version'", ".csv", "", .)

            qui{
                    ** Split pre quarters
                    split pre_quarters, parse(",")
                    drop pre_quarters
                    foreach var of varlist pre_quarters* {
                        replace `var' = subinstr(`var', "]", "", .)
                        replace `var' = subinstr(`var', "[", "", .)
                        replace `var' = subinstr(`var', ",", "", .)
                        replace `var' = subinstr(`var', "'", "", .)
                        replace `var' = strtrim(`var')
                    }

                    ** Reshape
                    reshape long pre_quarters, i(treated* control* *dist* lambda) j(aux)
                    egen baseline_period_length = max(aux)
                    drop aux 
                    
                    rename pre_quarters quarter
                    replace quarter = strtrim(quarter)
                    rename quarter quarter_str

                    ** Convert to Stata quarterly date
                    gen quarter = quarterly(quarter_str, "YQ")
                    drop quarter_str
                    format quarter %tq
                }

                ** Expand until 2024q4
                bys lam treated_id control_id: egen latest_quarter = max(quarter)
                preserve
                    ** Keep the latest quarter
                    keep if quarter == latest_quarter

                    ** Calculate the number of extra quarters to add (2024q4 = 259)
                    gen expand = 259 - latest_quarter

                    ** Expand
                    expand expand
                    drop expand 
 
                    ** Replace quarters
                    bys lam treated_id control_id: replace quarter = quarter + 1 if _n == 1 
                    bys lam treated_id control_id: replace quarter = quarter[_n-1] + 1 if _n > 1

                    drop latest_quarter
                    bys lam treated_id control_id: egen latest_quarter = max(quarter)
                    assert latest_quarter == 259

                    tempfile expanded
                    save "`expanded'"
                restore
                append using `expanded'
                drop latest_quarter

                ** Check that the reshape worked
                qui sum baseline_period_length, d 
                assert regexm("`sample_version'", "`r(max)'q")

                ** clean
                dropmiss, force 
                dropmiss, obs force
                keep treated_id control_id lambda_val quarter

                ** Filename 
                *gen sample_name = "`file'"

                ** Merge treated citations
                merge m:1 treated_id quarter using "`treated'", assert(1 2 3) keep(1 3) nogen
                replace treated_count = 0 if treated_count == .

                ** Merge control citations
                merge m:1 control_id quarter using "`control'", assert(1 2 3) keep(1 3) nogen
                replace control_count = 0 if control_count == .

                ** Merge Patents info
                tostring treated_id, replace
                replace treated_id = strtrim(treated_id)

                merge m:1 treated_id using "`patents'", assert(2 3) keep(3) nogen

                ** Progress
                local counter = `counter' + 1

                ** Save into Drive directly
                gisid lambda treated_id control_id quarter

                ** clean and save
                dropmiss, force 
                dropmiss, obs force

                compress
                order lambda treated_id control_id quarter  treated_count control_count   deal_id ult_parent acq_type acq_date
                sort lambda_val treated_id control_id quarter

                export delimited using "${out}\01 Sample final for ME - `sample_version' tr.csv", replace

            di "Processed `counter' files out of  `: word count `files''"
        }
