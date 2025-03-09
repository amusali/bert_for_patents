
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
* B. Load downloaded SDC Platinum data
* ==============================================================================
    ** Load
    mkf sdc 
    cwf sdc 

    import excel using "${raw}\Deals with treated assignees - 2.xlsx", firstrow clear

    ** Clean Ultimate Parent variable
    drop if  AcquirorUltimateParent == "HTC Corp"
    replace AcquirorUltimateParent = "GOOG" if AcquirorUltimateParent == "Alphabet Inc"
    replace AcquirorUltimateParent = "AMZ" if AcquirorUltimateParent == "Amazon.com Inc"
    replace AcquirorUltimateParent = "MSFT" if AcquirorUltimateParent == "Microsoft Corp"
    replace AcquirorUltimateParent = "FB" if inlist(AcquirorUltimateParent, "Meta Platforms Inc", "Facebook Inc")
    replace AcquirorUltimateParent = "APPL" if regexm(AcquirorUltimateParent, "Apple")


    ** Drop minority stake deals
    drop if regexm(DealSynopsis, "minority stake") & !regexm(DealSynopsis, "majority") 

    ** Drop previously crated vars
    drop Assignee* _merge AU Source Google
    duplicates drop

    ** Drop duplicates targets
    duplicates tag TargetFullName, gen(dup)
    preserve
        keep if dup 
        
        drop if yofd(dofc(DateAnnounced)) == 2000 // 5% stake only , see: https://press.aboutamazon.com/2000/1/amazon-com-to-buy-5-of-audible-inc-and-enters-strategic-alliance-allowing-customers-to-access-spoken-audio-from-audible-com-through-amazon-com
        drop if _n == _N // duplicate observation of Microsoft - WebTV deal

        tempfile aux
        save "`aux'"
    restore

    drop if dup
    append using `aux'
    gisid TargetFullName

    compress 
    save "${dta}/03 SDC data.dta", replace

    /*
    preserve
        keep TargetFullName
        gen id= _n 

        gen sub = strlower(substr(TargetFullName, 1, 2))

        tempfile sdc
        save "`sdc'"
    restore

    


********************************************************************************
* C. Load patents data and get assignees
********************************************************************************
    mkf pat 
    cwf pat
    ** Load 
    use "${dl}\00 Patents\dta\01 Patent data - without citations.dta", clear 

    ** Keep assignees
    keep assignee*
    duplicates drop

    ** Reshape
    gen version = _n
    rename *_notdisamb notdisamb_*
    rename *_disamb disamb_*

    reshape long disamb_assignee notdisamb_assignee, i(version) j(k)
    drop if mi(dis)
    drop v k
    duplicates drop 
    sort d 
    gen num = _n

    tempfile pats
    save "`pats'"


* ==============================================================================
* D. Merge data
* ==============================================================================
    
    ** Match it
    frame copy sdc clean
    cwf sdc 
    mkf all
    forvalue i = 1/`=_N'{
        cap restore
        cwf sdc
        loc sub = sub[`i']

        cwf pat
        preserve
            keep if substr(strlower(d), 1, 2) == "`sub'"

            tempfile aux
            save "`aux'"
        restore

        cwf clean
        preserve
            keep if sub == "`sub'"
            matchit id  TargetFullName  using `aux', idusing(num) txtusing(disamb_assignee) time override threshold(0.75)
            
            tempfile all
            save "`all'"
        restore
        
        frame all: append using `all'
    }
    *matchit num  disamb_assignee using `sdc', idusing(id) txtusing(TargetFullName) time override