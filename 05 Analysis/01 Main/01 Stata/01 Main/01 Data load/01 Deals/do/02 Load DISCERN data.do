
* ==============================================================================
* A. Set paths
* ==============================================================================
    clear all

    ** set path
    gl analysis "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main"
    gl python "${analysis}/00 Python data"
    gl stata "${analysis}/01 Stata"
    
    gl dl "${stata}/01 Main/01 Data load"
    gl deals "${dl}/01 Deals"
        gl aux "${deals}/_aux"
        gl do  "${deals}/do"
        gl dta "${deals}/dta"
        gl out "${deals}/out"
        gl raw "${deals}/raw"
        gl temp "${deals}/temp"

* ==============================================================================
* B. Load DISCERN subsidiary data
* ==============================================================================

    ** load data
    mkf deals
    cwf deals 

    use "${dta}/discern/discern_sub_names.dta", clear
    append using "${dta}/discern/discern_uo_names.dta"

    label var sample "U=UO+Labs+traded-Subs//S=SEC Exhibit 21"

    compress 
    save "${dta}/02 DISCERN - Acquisitions and subsidiaries.dta ", replace
    frame copy deals all


    ** Define GAFAM indicator for companies that became part of GAFAM at some point
    gen gafam = .
    forvalue i = 0/5{
        replace gafam = 1 if inlist(permno_adj`i', 90319, 10107, 84788, 14593, 13407  ) & mi(gafam) 
    }

    replace gafam = 0 if mi(gafam)
    keep if gafam 

    ** Remove internal subsidiaries of GAFAM
    drop if regexm(name_std, "GOOGLE") | regexm(name_std, "AMAZON") | regexm(name_std, "APPLE")  | /// 
    regexm(name_std, "FACEBOOK") | regexm(name_std, "MICROSOFT") | regexm(name_std, "META")  | regexm(name_std, "ALPHABET INC") 

    dropmiss, force
    drop *0
    
    ** Create ultimate parent and acquisition dates
    gen ult_parent = ""
    gen ult_parent_id = .
    gen acquisition_date = .
    gen resold = .
    gen resold_date = .
    gen target = ""
    gen target_id = .

    forvalue i = 0/3{
        **Keep a counter to run backwards in time
        local counter = 4 - `i'

        replace ult_parent = name_acq`counter' if mi(ult_parent) & !mi(name_acq`counter')
        replace ult_parent_id = permno_adj`counter' if mi(ult_parent_id) & !mi(permno_adj`counter')
        replace acquisition_date = fyear`counter' if mi(acquisition_date) & !mi(fyear`counter')

        
    }

    ** Keep if GAFAM 
    keep if inlist(ult_parent_id, 90319, 10107, 84788, 14593, 13407)

    replace target = name_std if inlist(permno_adj1, 90319, 10107, 84788, 14593, 13407)
    forvalue i = 1/3{
        local counter = 4 - `i'
        replace target = name_acq`counter' if mi(target) & !mi(name_acq`counter') & !inlist(permno_adj`counter', 90319, 10107, 84788, 14593, 13407)

    }

    drop if inlist(target, "INVESTOR GRP", "ALPHABET CAPITAL US LLC")
    drop if regexm(target, "MSFT")
    drop if target == "EBAY INC" // faulty row
    
    
    keep acquisition_date ult_parent ult_parent_id target country_code
    
    

    duplicates drop target, force
    
    cwf deals
    
    mkf clean
    
    forvalues j = 1/3{
        frame copy deals deals_helper`j'
        cwf deals
        *qui count
        forvalue i = 1/`=_N'{
            
            cwf deals
            local x = target[`i']
            local a = ult_parent[`i']
            di in red "Acquiror: `a'; Target:`x'"

            cwf all 
            cap restore
            preserve
                
                keep if name_acq`j' == "`x'"
                keep if sample == "S"

                if `=_N' == 0 {
                    continue
                }
                else{
                    
                    keep id_name name_std country fyear1 name_acq`j'
                    rename (id_name name_std country fyear1 name_acq`j') (acq_sub_id acq_sub_name acq_sub_country acq_sub_year target)

                    duplicates drop

                    tempfile aux
                    save "`aux'"
                }
                
        
            restore

            cwf deals_helper`j'
            merge 1:m target using `aux', assert(1 3) gen(merge`i'_`j')
            
            qui levelsof merge`i'_`j', loc(m)
            if `r(r)' == 2{
                preserve
                    keep if merge`i'_`j' == 3
                    *drop if acquisition_date < acq_sub_year & !mi(acq_sub_year)
                    tempfile s
                    save "`s'"
                    frame clean: append using `s'
                restore 
                keep if merge`i'_`j' == 1 

                drop acq_sub*
            }

        }
    }


    cwf deals 
    tempfile direct
    save "`direct'"

    cwf clean 
    append using `direct'

    ** Drop the subsidiaries of the target that has the same name (subs in other countries)
    split target
    drop if regexm(acq_sub_name, target1)
    drop merge* target?
    sort ult_parent_id target acquisition_date

    drop if !mi(acq_sub_country) & acq_sub_country != "US"
    drop if inlist(acq_sub_name, "MINORITY OWNED", "HALF OWNED")

    ** Save
    save "${temp}/02 Discern data - before regularization.dta", replace

********************************************************************************
* C. Identify resold date of GAFAm targets
********************************************************************************
    
    ** Load
    cap drop clean 
    mkf clean
    cwf clean
    use "${temp}/02 Discern data - before regularization.dta", clear

    drop if regexm(target, "PROPERTIES")
    keep if coun == "US" | mi(coun)

    gen child = acq_sub_name if !mi(acq_sub_name)
    replace chil = target if mi(child)

    bys target acq_sub_name (acq_sub_year): drop if _n == 2
    gisid child 

    
    ** Load patents
    mkf pat 
    cwf pat

    use "${dta}/discern/discern_pat_grant_1980_2021.dta", clear
    
    keep assignee_name name_std
    duplicates drop 
    rename name_std child 

    tempfile assignee_name
    save "`assignee_name'"

    cwf clean

    ** Merge with assignee names
    merge 1:m child using `assignee_name', keep(3)

    ** Motorola resold
    gen resold_date = 2014 if target == "MOTOROLA INC"
    drop if acq_sub_year > resold_date & !mi(acq_sub_year)

    drop _merge

    ** Ult parent
    replace ult_parent = "AMZ" if ult_parent_id == 84788
    replace ult_parent = "GOOG" if ult_parent_id == 90319
    replace ult_parent = "MSFT" if ult_parent_id == 10107
    replace ult_parent = "FB" if ult_parent_id == 13407
    replace ult_parent = "APPL" if ult_parent_id == 14593

    ** Save
    keeporder ult_parent acquisition_date target child acq_sub_name acq_sub_id acq_sub_year assignee_name
    rename acquisition_date acquisition_year

    sort ult_parent acquisition_year target child
    compress 

    save "${dta}/DISCERN - deals and assignees.dta", replace
