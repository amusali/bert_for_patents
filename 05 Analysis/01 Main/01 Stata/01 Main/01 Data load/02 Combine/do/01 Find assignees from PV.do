
* ==============================================================================
* A. Set paths
* ==============================================================================
    clear all

    ** set path
    gl analysis "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main"
    gl python "${analysis}/00 Python data"
    gl stata "${analysis}/01 Stata"
    
    gl dl "${stata}/01 Main/01 Data load"
    gl deals "${dl}/02 Combine"
        gl aux "${deals}/_aux"
        gl do  "${deals}/do"
        gl dta "${deals}/dta"
        gl out "${deals}/out"
        gl raw "${deals}/raw"
        gl temp "${deals}/temp"

* ==============================================================================
* B. Load downloaded matched datasets
* ==============================================================================
    ** Load

    local methods `" "tc" "bfl"  "'
    local samples `" "won" "wn"  "'


    foreach method of local methods{
        foreach sample of local samples{
            ** Load CSV
            import delimited using "${raw}/matched_`method'_`sample'.csv", clear

            ** rename
            rename m match_`method'_`sample'

            if "`method'" == "tc"{
                rename sim sim_`method'_`sample'
            }
            else{
                rename fin sim_`method'_`sample'
            }

            ** checks
            tempfile `method'_`sample'
            save "``method'_`sample''"
        }
    }
   
    ** Combine the datasets
    local i = 1
    foreach method of local methods{
            foreach sample of local samples{

                if `i' == 1{
                    ** Load CSV
                    use ``method'_`sample'', clear
                }
                else{
                    merge 1:1 small_dataset_name using ``method'_`sample'', assert(3) nogen
                }

                local i = `i' +1
                

            }
        }

********************************************************************************
* C. Classify and clean matches 
********************************************************************************
    *---------------------------------------------------------*
    * C.1. Same matches
    *---------------------------------------------------------* 
        mkf clean
        gen same = match_tc_wn == match_tc_won & match_bfl_wn == match_bfl_won & match_bfl_wn == match_tc_wn
        preserve
            keep if same 

            ** Keep one variable for a match
            keep small_dataset_name match_tc_won

            rename (small_dataset_name match_tc_won) (child assignee)

            tempfile same
            save "`same'"

            frame clean: append using `same'
        restore

        drop if same 
        drop same
    
    *---------------------------------------------------------*
    * C.2. No matches
    *---------------------------------------------------------*
        replace sim_tc_won = 0 if match_tc_won == "A & V, Inc."
        gen none = sim_tc_won + sim_tc_wn + sim_bfl_wn + sim_bfl_won == 0 
        tab none

        preserve
            keep if none 
            keep small_dataset_name

            frame copy default none

            ** Export to excel
            gen tokeep = .

            export excel using "${aux}/00 No matches.xlsx", firstrow(variables) cell("B2") sheet("out", replace)
        restore

        drop if none 
        drop none

    *---------------------------------------------------------*
    * C.3. One time matches
    *---------------------------------------------------------*
        gen aux = sim_tc_won + sim_tc_wn + sim_bfl_wn + sim_bfl_won
        gen one = 0
        foreach var of varlist sim*{
            replace one = 1 if aux == `var'
        }

        preserve
            keep if one

            ** COmbine information in the variables
            reshape long match sim, i(small) j(v) str
            gsort -sim
            drop if sim == 0
            keep small match 
            duplicates drop 
            
            ** Export to Excel file to check manually
            gen tokeep = .

            export excel using "${aux}/01 One-time matches.xlsx", firstrow(variables) cell("B2") sheet("out", replace)
        restore
        
        drop if one 
        drop one aux

    *---------------------------------------------------------*
    * C.4. Matches of one TF-IDF with Cosine Sim only
    *---------------------------------------------------------*
        preserve
            keep if match_bfl_wn == match_bfl_won & match_bfl_won == "NO MATCH"

            drop *bfl*
            duplicates drop
            reshape long match sim, i(small) j(v) str
            gsort -sim
            drop if sim == 0
            duplicates drop small match, force

            ** Export to Excel file to check manually
            gen tokeep = .
            export excel using "${aux}/02 Matches of one TF-IDF.xlsx", firstrow(variables) cell("B2") sheet("out", replace) 
        restore

        drop if match_bfl_wn == match_bfl_won & match_bfl_won == "NO MATCH"

    *---------------------------------------------------------*
    * C.5. Matches of one BFL only
    *---------------------------------------------------------*
        preserve
            keep if sim_tc_won == sim_tc_wn & sim_tc_wn == 0

            drop *_tc*
            reshape long match sim, i(small) j(v) str
            gsort -sim
            drop if sim == 0
            duplicates drop small match, force 

            ** Export to Excel file to check manually
            gen tokeep = .
            export excel using "${aux}/03 Matches of one BFL.xlsx", firstrow(variables) cell("B2") sheet("out", replace)
            
        restore

        drop if sim_tc_won == sim_tc_wn & sim_tc_wn == 0

    *---------------------------------------------------------*
    * C.6. The rest of the matches
    *---------------------------------------------------------*
        reshape long match sim, i(small) j(v) str
        gsort -sim
        duplicates drop small match, force 

        ** Export to Excel file to check manually
        gen tokeep = .
        export excel using "${aux}/04 Rest of the matches.xlsx", firstrow(variables) cell("B2") sheet("out", replace) 

********************************************************************************
* D. Bring back all the checked matches
********************************************************************************
    cwf clean 
    
    preserve
        import excel using "${aux}/01 One-time matches.xlsx", firstrow sheet("in") clear


        rename (small_dataset_name match) (child assignee)
        keep if tokeep == 1
        drop tokeep

        tempfile data1
        save "`data1'"
    restore

    preserve
        import excel using "${aux}/02 Matches of one TF-IDF.xlsx", firstrow sheet("in") clear


        rename (small_dataset_name match) (child assignee)
        keep if tokeep == 1
        drop tokeep


        tempfile data2
        save "`data2'"
    restore

    preserve
        import excel using "${aux}/03 Matches of one BFL.xlsx", firstrow sheet("in") clear


        rename (small_dataset_name match) (child assignee)
        keep if tokeep == 1
        drop tokeep


        tempfile data3
        save "`data3'"
    restore

    append using `data1'
    append using `data2'
    append using `data3'

    drop sim 
    duplicates drop 
    gisid child

    ** Save
    save "${dta}/01 Matches assignees.dta", replace

********************************************************************************
* D. Combine back with the deals data
********************************************************************************
    ** Load deals data 
    use "${dta}/06 All deals.dta", clear

    ** Combine with deals data
    merge m:1 child using "${dta}/01 Matches assignees.dta", assert(1 3)

    ** remove the unmatched deals
    preserve
        keep if _merge == 1
        compress 

        save "${dta}/01 Unmatched deals.dta", replace
    restore

    keep if _merge == 3
    drop _merge

    ** Drop old assignee variable
    drop assignee_name
    duplicates drop

    ** Drop deals that were not full acquisitions // wrong matches
    drop if regexm(deal_synopsis, "stake") & !inlist(child, "WEBTV NETWORKS INC", "NEXT COMPUTER INC" )
    drop if assignee == "Nuance Corporation, Inc."
    drop if child == "SYMBOL TECH INC" // went with Motorola Solutions, not with Motorola Mobility


    ** Create a Deal ID
    drop deal_id // coming from GAFAM empire dataset
    egen deal_id = group(ult_parent assignee)

    ** Add extra matches based on manual research
    *** Activision Blizzard
    preserve
        keep if deal_id == 207
        insobs 1
        replace assignee = "Activision Publishing, Inc." if mi(deal_id)

        insobs 1
        replace assignee = "BLIZZARD ENTERTAINMENT, INC." if mi(deal_id)

        replace deal_id = 207
        replace acquisition_date = td(12oct2023)
        replace ult_parent = "MSFT"

        tempfile activision
        save "`activision'"
    restore
    drop if deal_id == 207
    append using `activision'


    *** Shazam Entertainment
    preserve
        keep if deal_id == 81
        insobs 1
        replace assignee = "Shazam Investments Ltd." if mi(deal_id)

        replace deal_id = 81
        replace acquisition_date = td(11dec2017)
        replace ult_parent = "APPL"

        tempfile shazam
        save "`shazam'"
    restore
    drop if deal_id == 81
    append using `shazam'

 ********************************************************************************
 * E. Fix dates
 ********************************************************************************
    cap drop dup
    duplicates tag deal_id, gen(dup)
    bys deal_id (acquisition_date): gen check = 1 if acquisition_date[2] == . & dup == 1
    bys deal_id (acquisition_date): replace acquisition_date = acquisition_date[1] if check == 1 & _n == 2 
    drop check dup 

    *** Whatsapp case
    replace acquisition_date = td(5oct2014) if deal_id == 116

    *** Fitbit case
    replace acquisition_date = td(13jan2021) if deal_id == 148

    *** LinkedIn case
    replace acquisition_date = td(8dec2016) if deal_id == 264

    *** Motorola case
    replace acquisition_date = td(22may2012) if regexm(child, "MOTOROLA") | regexm(parent, "MOTOROLA")

    *** Anapurna Labs
    replace acquisition_date = td(22jan2015) if deal_id == 5

    *** Nuance Communications
    replace acquisition_date = td(3mar2022) if parent == "Nuance Communications"
/*
    *** Make SDC Platinum the main date of acquisition 
    duplicates tag deal_id, gen(dup)
    preserve
        keep if dup
        drop dup

        ** Create sdc tags
        bys deal_id: gen sdc = inlist(source, "SDC Platinum")
        bys deal_id: egen aux = max(sdc)
        bys deal_id: gen aux_date = acquisition_date if sdc == 1

        bys deal_id (aux_date): replace acquisition_date = aux_date[1] if aux == 1

        tempfile aux
        save "`aux'"
    restore
    keep if ~dup 
    append using `aux'

    assert !mi(acquisition_date)

    ** Acquisition year
    replace acquisition_year = yofd(acquisition_date)
 
    ** Save
    order source deal_id 
    compress
    save "${dta}/01 All deals - cleaned and matched.dta", replace



