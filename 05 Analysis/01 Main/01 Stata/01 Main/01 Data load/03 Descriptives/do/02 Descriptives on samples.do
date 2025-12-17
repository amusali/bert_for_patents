
* ==============================================================================
* A. Set paths
* ==============================================================================
    clear all

    ** set path
    gl analysis "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main"
    gl python "${analysis}/00 Python data"
    gl stata "${analysis}/01 Stata"
    
    gl dl "${stata}/01 Main/01 Data load"
    gl desc "${dl}/03 Descriptives"
        gl aux "${desc}/_aux"
        gl do  "${desc}/do"
        *gl dta "${desc}/dta"
        gl out "${desc}/out"
        gl raw "${desc}/raw"
        gl temp "${desc}/temp"
    gl dta "${stata}/01 Main/03 Estimation/03 CS-DiD/dta"

    ** Colours - source: https://lospec.com/palette-list/new-tableau-10
    global c_blue    "#557ca7"
    global c_orange  "#e98a2e"
    global c_red     "#d5525a"
    global c_teal    "#80b9b2"
    global c_green   "#66a14f"

    global c_yellow  "#e9c74c"
    global c_purple  "#aa7aa0"
    global c_pink    "#f79ba8"
    global c_brown   "#98745e"
    global c_gray    "#b9b0ab"

********************************************************************************
* B. Load grid results for citation
********************************************************************************
    /* ** Loop over filenames and load and append
    local types `" "M&A"  "Off deal" "'
    local tts `" "baseline" "top-tech" "'

    mkf clean 

    foreach type of local types {
        foreach tt of local tts{

            ** Adjust filename
            if "`tt'" == "baseline" {
                local filename = "01 Sample - `type', `tt', 4q, caliper0.0500 - all patents, for csdid - no exact matching on grant year.dta"
                }
            else if "`tt'" == "top-tech" {
                local filename = "01 Sample - `type', `tt', 80, 4q, caliper0.0500 - all patents, for csdid - no exact matching on grant year.dta"
            }

            ** Load
            use if cohort != 0 using "${dta}/`filename'", clear
            
            ** Drop lambda and duplicates
            drop lambda
            duplicates drop 
            gisid patent_id quarter

            ** Gen sample variable
            gen sample = "`type' - `tt'"

            ** Add into a new frame
            tempfile aux
            save "`aux'"
            frame clean: append using `aux'
            
        }
    }

    cwf clean 
    order sample 

    save "${dta}\02 All samples - acquired patents with citations.dta", replace */

********************************************************************************
* C. Descriptives on samples
********************************************************************************
    
    *---------------------------------------------------------*
    * C.1. Simple stats
    *---------------------------------------------------------*
    ** Load data with all samples
    use "${dta}\02 All samples - acquired patents with citations.dta", clear

    ** Age variable 
    gen age_in_acquisition = cohort - qofd(grant_date)
    assert age_in_acquisition >= 4 // min 4 quarters pre acquisition

    replace age_in_acquisition = age_in_acquisition / 4 // in years

    ** Filter
    keeporder sample ult_parent deal_id patent_id age* cpc_subclass_current num_claims
    duplicates drop
    gisid sample patent_id

    ** Add total 
    preserve
        replace ult_parent = "Total"

        tempfile total
        save "`total'"
    restore
    append using `total' 
    
    ** Encode CPC
    egen cpc_num = group(cpc_subclass_current) 

    ** Descriptives
    #delimit ;
        gcollapse 
        (nunique) num_patents = patent_id 
        (nunique) num_deals = deal_id 
        (mean) avg_age_at_acq = age_in_acquisition
        (sd) sd_age_at_acq = age_in_acquisition
        (nunique) num_cpc_subclasses = cpc_num
        (mean) avg_num_claims = num_claims
        ,by(sample ult_parent)
    ;                     
    #delimit cr
    replace num_deals = . if regexm(sample, "Off deal") // add the figure manually 

    gisid sample ult_parent
    tempfile simple_stats
    save "`simple_stats'"

    ** Export table 

    *** Digits
    qui ds sample ult_parent, not 
    foreach var in `r(varlist)'{
        replace `var' = round(`var', 0.01)
    }

    *** Make a matrix first
    mkmat num_patents num_deals avg_age_at_acq sd_age_at_acq num_cpc_subclasses avg_num_claims, matrix(desc_mat) 
    
    *** Assign row names
    local rn
    quietly {
        forvalues i = 1/`=_N' {
            local rn `"`rn' `=ult_parent[`i']'"'
        }
    }
    matrix rownames desc_mat = `rn'

    *** Assign column names
    matrix colnames desc_mat = "# patents" "# deals" "Avg age at acquisition" "SD age at acquisition" "# CPC subclasses" "Avg num claims"
    esttab matrix(desc_mat) using "${out}\02. Simple stats.tex", replace tex

/*
*---------------------------------------------------------*
* C.2. Avg citation before and after acquisition
*---------------------------------------------------------*
    ** Load data with all samples
    use "${dta}\02 All samples - acquired patents with citations.dta", clear

    ** Filter age 
    gen age = quarter - qofd(grant_date)
    keep if age <= 80 // 20 years max allowed age
    assert age >= 0 

    ** Tag before/after 
    gen after = quarter >= cohort

    ** Drop patents acquired after the age of 20 years 
    bys sample ult_parent patent_id: egen acquired_before_age20 = max(after)
    drop if acquired_before_age20 == 0
    drop acquired_before_age20

    ** Filter
    keeporder sample ult_parent patent_id citation after
    gcollapse (mean) citation, by(sample ult_parent patent_id after)

    tab after 

    ** Add total 
    preserve
        replace ult_parent = "Total"

        tempfile total
        save "`total'"
    restore
    append using `total' 
 
    ** Average over patents
    gcollapse (mean) avg_cit = citation (sd) sd_cit = citation (median) median_cit = citation (p20) p20_cit = citation (p80) p80_cit = citation, by(sample ult_parent after)

    ** Reshape
    reshape wide avg_cit sd_cit median_cit p20_cit p80_cit, i(sample ult_parent) j(after)
    rename *0 *_before 
    rename *1 *_after

    ** Merge with simple stats
    merge 1:1 sample ult_parent using "`simple_stats'", assert(3) nogen

    order sample ult_parent num_patents num_deals ///
        avg_age_at_acq sd_age_at_acq num_cpc_subclasses avg_num_claims ///
        avg_cit_before sd_cit_before median_cit_before p20_cit_before p80_cit_before ///
        avg_cit_after sd_cit_after median_cit_after p20_cit_after p80_cit_after

/*
    ** Export table
    estpost tabstat num_patents avg_age_at_acq sd_age_at_acq num_cpc_subclasses avg_num_claims ///
        avg_citations_pre_acq sd_citations_pre_acq ///
        avg_citations_post_acq sd_citations_post_acq ///
        , by(sample) statistics( count mean sd min p25 p50 p75 max )
/*
    esttab ., cells("count(fmt(0)) mean(fmt(2)) sd(fmt(2)) min(fmt(0)) p25(fmt(0)) p50(fmt(0)) p75(fmt(0)) max(fmt(0))") ///
        label noobs nomtitles nonumber compress ///
        title("Descriptive statistics on acquired patents with citations, by sample") ///
        alignment(D{.}{.}{-1}) ///
        mgroups(" " , pattern(1 0 0 0 0 0 0 0 0)) ///
        replace ///
        using "${out}\desc\02 Descriptives on samples - acquired patents with citations.rtf"


