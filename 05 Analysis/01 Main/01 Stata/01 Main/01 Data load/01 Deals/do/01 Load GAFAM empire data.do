
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

* ==============================================================================
* B. Load GAFM empire data
* ==============================================================================

    ** load data
    import excel using "${raw}\GAFAM empire deals.xlsx", clear firstrow

    ** Clean
    dropmiss, force
    dropmiss, obs force
    duplicates drop

    ** Drop GAFAm themselves
    duplicates tag child, gen(dup)
    drop if dup == 0 & count == "2" // drops 5 rows

    *---------------------------------------------------------*
    * B.1. Remove resold targets, name duplicates, and Google Cloud Platform
    *---------------------------------------------------------*
    
        ** remove cloud platforms
        duplicates tag child parent, gen(aux)
        drop if aux == 1
        drop aux
        gisid child parent

        ** Extract the year
        gen len = strlen(dateparent)

        gen pdate = date(dateparent, "DMY") if len == 9
    
        format pdate %td 
        order child parent ult date pdate
        
        bys child: egen max_pdate = max(pdate) if pdate != .
        bys child: egen max_date = max(date) if date != .
        format max* %td
        order child parent ult date pdate max_pdate max_date

        preserve
            ** Keep if duplicates
            keep if dup 

            keep child max_pdate max_date dup count
            drop if mi(max_pdate)
            duplicates drop

            keep if max_pdate  < max_date

            gen resold = 1

            gisid child 

            tempfile resold
            save "`resold'"
        restore

        merge m:1 child using "`resold'", assert(1 3) nogen

        ** Determine each reselling of targets individually
        preserve
            ** Filter and save
            keep if resold == 1
            drop resold
            
            gen resold = . // 1 - resold, 0 - not resold
            order resold 

            export excel using "${aux}\01 List of acquisitions that were resold.xlsx", firstrow(variables) sheet("out", replace) cell("B2") 
        restore

        ** Load back and keep the ones
        preserve
            import excel using "${aux}\01 List of acquisitions that were resold.xlsx", firstrow sheet("in") clear

            dropmiss, force

            tempfile resold_checked
            save "`resold_checked'"
        restore

        drop resold
        merge 1:1 child parent using `resold_checked', assert(1 3) 
        
        ** Drop rows
        drop if todrop == 1
        drop todrop
        gen resold_date =  max_date if !mi(resold) & _merge == 3
        drop max*
        format resold_date %td
        assert !mi(resold_date) if !mi(resold)
    
    *---------------------------------------------------------*
    * B.2. Drop other wrong acquisitions and harmonize Ultime Parent (GAFAM) name
    *---------------------------------------------------------*
        ** Drop
        drop if child == "Redwood Robotics" & parent == "Apple" // faulty row
        replace ult_parent = "GOOG" if child == "Redwood Robotics" & parent == "Google"
        replace ult_parent = "AMZ" if child == "Songza" & parent == "Amie Street"
        replace ult_parent = "GOOG" if child == "Songza" & parent == "Google"
        replace resold = . if child == "Songza" & parent == "Google"
        replace resold_date = . if child == "Songza" & parent == "Google"

        drop if parent == "Alphabet" & child == "Nest Labs"
        drop if child == "Facebook Careers"
        drop if child == "Topspin Media" 
        drop if child == "Undead Lab"
        drop if child == "QuadraMed"
        drop if child == "Back To Basics Toys"
        *drop if child == "Atmel"
        drop if child == "280 North"
        drop if child == "ClickEquations" 
        drop if regexm(ustrlower(child), "amazon")

        replace child = "Motorola Mobility" if child == "Google, Motorola Mobility division"
        drop if strlower(substr(child), 1, 6) == "google"
        drop if strlower(substr(child), 1, 8) == "facebook" 
        drop if child == "Workplace by Facebook"

        replace pdate = . if parent == "Amazon Web Services"
        replace pdate = . if parent == "Meta"
        replace parent = "Amazon" if substr(parent, 1, 6) == "Amazon"
        replace parent = "Google" if substr(parent, 1, 6) == "Google" | parent == "X: Alphabet’s moonshot factory"
        replace parent = "Google" if parent == "Alphabet"

        replace pdate = . if inlist(parent, "Google", "Amazon")
        replace parent = "Motorola Mobility" if parent == "Google, Motorola Mobility division"

        drop if child == parent

        ** Check the number of Ultime Parents
        qui levelsof ult_parent, loc(gafam_firms)
        assert `r(r)' == 5 


    *---------------------------------------------------------*
    * B.3. Identify subsidiaries that were acquired before & after GAFAM acquired a target
    *---------------------------------------------------------*

        ** Before & after GAFAM
        gen before = date < pdate if !mi(pdate)

        sort ult_parent parent date 

        ** Clean acquisition date variable
        gen related_acq_date = pdate if !(inlist(parent, "Meta", "Google Cloud", "Google for Startups") | regexm(ustrlower(parent), "amazon"))
        replace related_acq_date = date if mi(pdate)
        format related_acq_date %td


    *---------------------------------------------------------*
    * B.4. Drop the rows for the past acquiistion history of target companies
    *---------------------------------------------------------*
        ** Drop the past acqisitions of GAFAM targets as they are irrelevant
        gen counter = dateparent == "#N/A"
        bys child : egen aux = max(counter)
        bys child (date): gen todrop = aux == 1 & dateparent[1] == "#N/A"
        bys child: drop if _n == 1 & todrop == 1
        bys child (date): drop if _N > 1 & _n == 1 & dateparent[1] == "#N/A"

        ** Renew duplicate variable based on child

        bys child (date): drop if _n == 3 
        drop dup
        duplicates tag child, gen(dup)

        ** update resold information
        replace resold = 1 if resold != 1 & dup
        bys child (date): replace resold_date = date[2] if mi(resold_date)
        replace resold = . if dup == 0 | mi(resold)

        ** Update resold info for complex deals 
        drop if inlist(child, "Atmel") | inlist(parent, "Atmel") 
        replace resold_date = td(8feb2021) if inlist(child, "Dialog Semiconductor", "Atmel") | inlist(parent, "Dialog Semiconductor", "Atmel") 
        replace resold = 1 if inlist(child, "Dialog Semiconductor", "Atmel") | inlist(parent, "Dialog Semiconductor", "Atmel") 

        replace resold_date = . if inlist(child, "GamerDeal.TV", "MetaWatch")
        replace resold = . if inlist(child, "GamerDeal.TV", "MetaWatch")

        ** Drop rows where target is resold
        drop if date == resold_date

********************************************************************************
* C. Identify and group deals
********************************************************************************

    ** Fill acquiistion date variable
    replace related_acq_date = date if mi(related_acq_date)

    gen acquisition_date = related_acq_date
    replace acquisition_date = date if date > related_acq_date & !mi(date)
    format acquisition_date %td

    ** Deal synopsis
    gen deal_synopsis = "Direct GAFAM acquisition" if len == 19 | /// 
    regexm(strlower(parent), "amazon") | strlower(parent) == "meta" | regexm(strlower(parent), "google") ///
    | regexm(strlower(parent), "alphabet") | regexm(strlower(parent), "apple") | regexm(strlower(parent), "microsoft")

    replace deal_synopsis = "Acquired as subsidiary of GAFAM target" if before == 1
    replace deal_synopsis = "Acquired as a child of target post-acquisition by GAFAM" if before == 0
    replace deal_synopsis = deal_synopsis + "; resold after acquisition" if resold == 1
    assert !mi(deal_synopsis)
    ** Sort
    sort ult_parent acquisition_date date related_acq_date  
    order ult_parent related_acq_date acquisition_date date parent child before deal_synopsis resold*

    ** Define deal ID
    egen id = group(ult_parent acquisition_date)
    bys id: gen diff = acquisition_date == date
    bys id (date): gen sum = diff[1]
    bys id (date): replace sum = diff + sum[_n-1] if _n > 1
    by id: gen helper = sum[_N] == _N if _N>1
    egen iid = group(id sum) if helper == 1
    tostring id iid, replace

    ** Group id
    egen  x = group(id iid)
    order x 
    rename x deal_id
    drop id iid helper sum diff dup

    ** Duplicates at ID x date level (4 observations)
    duplicates tag deal_id date, gen(dup)
    br if dup 
    tostring deal_id, replace
    replace deal_id = "999" + deal_id if inlist(child, "Gizmofive", "Fastlane")
    egen x= group(deal_id), autotype
    drop deal_id
    rename x deal_id

********************************************************************************
* D. Clean and save
********************************************************************************
    ** Vars to keep 
    keeporder deal_id ult_parent related_acq_date acquisition_date date parent child before deal_synopsis resold resold_date ///
    total_products_active industry industry_group loc full_descrip descrip patents_granted /// 
    top_5_investors num_of_investors patents_granted trademarks_registered most_popular_trademark_class child_link parent_link loc_city loc_state AN industry*

    rename AN country
    dropmiss, force

    compress
    gisid deal_id date 
