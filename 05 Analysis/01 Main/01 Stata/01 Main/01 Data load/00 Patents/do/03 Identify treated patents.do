
* ==============================================================================
* A. Set paths
* ==============================================================================

    clear all

    ** Set paths
    gl analysis "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main"
    gl python "${analysis}/00 Python data"
    gl stata "${analysis}/01 Stata"
    
    gl dl "${stata}/01 Main/01 Data load"
        gl do "${dl}/do"
        gl dta "${dl}/dta"
        gl out "${dl}/out"
        gl raw "${dl}/raw"

* ==============================================================================
* B. Load treated assignees data
* ==============================================================================
    ** Load
    mkf assignees 
    cwf assignees 

    use "${dta}\02 Deals with treated assignees.dta", clear

    ** Clean
    duplicates drop
    duplicates tag Assignees, gen(dup)

    preserve
        ** Will deal with this later !!!
        keep if dup > 0

        tempfile duplicated_assignees
        save "`duplicated_assignees'"
    restore
    keep if dup == 0
    drop dup
    gisid Assignees
    format Assignees %75s

    tempfile assignees
    save "`assignees'"

    ** Load processed patents data
    mkf patents
    cwf patents

    use "${dta}\01 Patent data - without citations.dta", clear

* ==============================================================================
* C. Combine two datasets based on Assignee information
* ==============================================================================

    ** Define locals
    local ambiguity `" "disamb" "notdisamb" "'
    
    mkf all 
    foreach variation of local ambiguity{
        forvalue assignee_sequence = 0/4{
            preserve

                di in red "Processing variable assignee`assignee_sequence'_`variation' "

                ** Keep patent ID and the relevant assignee variable
                keep patent_id assignee`assignee_sequence'_`variation'
                drop if mi(assignee`assignee_sequence'_`variation')

                ** Rename assignee variable
                rename assignee`assignee_sequence'_`variation' Assignees
                drop if strlen(Assignees) > 99
                gen aux = Assignees 
                drop Assignees
                rename aux Assignees
                *format Assignees %99s
                    
                ** Merge with deals data 
                merge m:1 Assignees using `assignees', keep(3) nogen 

                ** Update note
                gen note = "Matched on assignee`assignee_sequence'_`variation'"

                ** Save
                tempfile matches
                save "`matches'"

                ** Write to another frame 
                frame all: append using `matches'
            restore
        }

    }


* ==============================================================================
* D. Combine treated patents with the processed data to create a working sample
* ==============================================================================
    cwf all

    ** clean
    duplicates drop patent_id, force 
    assert !mi(Assignees)

    ** Save
    tempfile treated
    save "`treated'"

    ** Combine back to original data
    cwf patents
    merge 1:1 patent_id using `treated', assert(1 3) keep(3) nogen

    ** Create two samples - before and after acquisition
    gen grant_date = date(patent_date, "YMD")
    format %td grant_date
    gen before_acquisition = grant_date <= DateEffective if !mi(DateEffective)
    gen after_acquisition = grant_date > DateEffective if !mi(patent_date)

    ** Save both samples 
    preserve
        keep if before_acquisition

        compress 
        save "${dta}\03 Treated patents - before acquisition.dta", replace
    restore

    preserve
        keep if after_acquisition

        compress 
        save "${dta}\03 Patents of acquired firms - after acquisition.dta", replace
    restore



    