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
    replace config = strtrim(subinstr(config, "01 Hybrid matches - ", "", .))
    replace config = strtrim(subinstr(config, "01 Hybrid matches (lambda 0.6 and 0.7) - ", "", .))
    
    destring id, replace
    compress
    tostring lam, replace
    replace lam = subinstr(lam, ".", "", .)

* ==============================================================================
* C. Loop over configs x treated flag x lambdas to calculate overlaps
* ==============================================================================

    ** Locals
    levelsof file_id, local(files)
    levelsof lambda_val, local(lambdas)
    foreach file of local files{
        foreach lam of local lambdas{
            preserve
                keep if lam == "`lam'" & file_id == `file'
                gen count = _N

                tempfile lam_`lam'_`file'
                save "`lam_`lam'_`file''"
            restore
        }
    }
    
    foreach file of local files{
        foreach lam of local lambdas{
            foreach lam1 of local lambdas{
                ** Skip if the lambdas are the same
                if "`lam'" == "`lam1'" continue
                
                preserve
                    u `lam_`lam'_`file'', clear
                    merge 1:1 id using `lam_`lam1'_`file'', gen(overlap)

                    gcollapse (nunique) id, by(overlap file config treated)
                    gen lam = "`lam'"
                    gen lam1 = "`lam1'"

                    tempfile lam_`lam'_`lam1'_`file'
                    save "`lam_`lam'_`lam1'_`file''"
                restore
    
            }
        }
    }
    clear

    foreach file of local files{
        foreach lam of local lambdas{
            foreach lam1 of local lambdas{
                if "`lam'" == "`lam1'" continue
                
                append using "`lam_`lam'_`lam1'_`file''"
    
            }
        }
    }
    save "${dta}\04 Overlaps - all.dta", replace
* ==============================================================================
* D. Reconcile and finalize overlap figures in %
* ==============================================================================
    u "${dta}\04 Overlaps - all.dta", clear
    rename id count

    ** Drop the mirrored copies of overlaps 
    drop if lam > lam1

    ** Filter relevant overlaps for paper
    keep if regexm(config, "4q") & regexm(config, "caliper_0.0500")
    drop if lam == "6" & lam1 == "7"
    drop if lam1 == "6" & regexm(config, "Off deal")
    drop if lam1 == "7" & regexm(config, "M&A")

    ** Sort
    order config treated lam lam1 count overlap
    sort config treated lam lam1 overlap

    /* ** Calculate Total Counts
    bys config treated lam lam1 (overlap): gen tc = count[1] + count[3]
    bys config treated lam lam1 (overlap): gen tc1 = count[2] + count[3]

    bys config treated lam lam1 (overlap): replace tc = count[1] + count[2] if overlap[1] == 1 & mi(tc)
    bys config treated lam lam1 (overlap): replace tc1 = count[1] + count[2] if overlap[1] == 2 & mi(tc)

    bys config treated lam lam1 (overlap): replace tc = count[2] if mi(tc)
    bys config treated lam lam1 (overlap): replace tc1 = count[2] if mi(tc1)

    assert !mi(tc) & !mi(tc1) */

    ** Calculate Jaccard index
    gcollapse (sum) union = count, by(config treated lam lam1) merge replace
    preserve
        keep if overlap == 3
        gen jaccard = count / union

        tempfile jaccard
        save "`jaccard'"
    restore

    merge m:1 config treated lam lam1 using `jaccard', assert(3) nogen