* ==============================================================================
* A. Set paths
* ==============================================================================
    clear all
    capture log close
    set more off
    set type double, perm
    set excelxlsxlargefile on

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
        gl out "${csdid}\out\paper"
        gl raw "${csdid}\raw"
        gl temp "${csdid}\temp"
        gl log "${csdid}\log"

    gl raw_drive "${google_drive}\PhD Data\12 Sample Final\actual results\citation_noexactmatch_on_grantyear"
    gl pca_drive "${google_drive}\05 Analysis\01 Main\00 Python data\01 CLS embeddings"

    gl matches "${google_drive}\PhD Data\11 Matches\actual results"

    global c_blue    `=hextorgb("#557ca7")'
    global c_orange  `=hextorgb("#e98a2e")'
    global c_red     `=hextorgb("#d5525a")'
    global c_teal    `=hextorgb("#80b9b2")'
    global c_green   `=hextorgb("#66a14f")'

    global c_yellow  `=hextorgb("#e9c74c")'
    global c_purple  `=hextorgb("#aa7aa0")'
    global c_pink    `=hextorgb("#f79ba8")'
    global c_brown   `=hextorgb("#98745e")'
    global c_gray    `=hextorgb("#b9b0ab")'

    qui hextorgb, hex("#557ca7" "#d5525a" "#f79ba8" "#b9b0ab" ) // blue, red, pink, gray
    global c_blue "`r(rgb1)'"
    global c_red "`r(rgb2)'"
    global c_pink "`r(rgb3)'"
    global c_gray "`r(rgb4)'"


* ==============================================================================
* B. Load and clean and plot for non-PC covariates
* ==============================================================================
   /*
    ** Load
    u "G:\My Drive\uc3m PhD\05 Analysis\01 Main\01 Stata\01 Main\04 Matching quality\out\02. SMDs.dta", clear

    ** Filter and clean
    drop if regexm(cov, "pc")
    keep if inlist(lambda, 0, 1, 99) | (lambda == 0.6 & acq_type == "M&A") | (lambda == 0.7 & acq_type == "Off deal")
    tostring lam, replace
    replace lam = subinstr(lam, ".", "0", .)
    replace smd = abs(smd)

    gen sort = .
    replace sort = 1 if cov == "cit_m1"
    replace sort = 2 if cov == "cit_m2"
    replace sort = 3 if cov == "cit_m3"
    replace sort = 4 if cov == "cit_m4"
    replace sort = 5 if cov == "age"
    replace sort = 6 if cov == "num_claim"

    gen aux = acq + ", " + bl

    
    ** Setup
    egen id = group(acq bl), label
    qui levelsof id, local(ids)

    local m_lambda0   "msymbol(circle)      mcolor($c_blue)   mfcolor($c_blue) msize(large)"
    local m_lambda_opt  "msymbol(diamond)     mcolor($c_red)    mfcolor($c_red) msize(large)"
    local m_lambda1   "msymbol(triangle)    mcolor($c_pink)  mfcolor($c_pink) msize(large)"
    local m_before    "msymbol(square)      mcolor($c_gray) mfcolor($c_gray) msize(large)"

     foreach sample of local ids{
            preserve
                ** Filter
                keep if id == `sample'
                local sam = aux[1]

                ** Reshape
                reshape wide smd, i(id covariate) j(lam) str
                dropmiss, force
                ds

                ** Optimal lambda
                if regexm("`sam'", "M&A"){
                    local opt_lam_text = `"{it:{&lambda} = 0.6}"'
                }
                else{
                    local opt_lam_text = `"{it:{&lambda} = 0.7}"'
                }

                #delimit ;
                    graph dot smd*, 
                    over(
                        covariate ,
                         sort(sort)
                        relabel(
                            2 `" "Pre-treatment" "citation (t-1)" "'
                            3 `" "Pre-treatment" "citation (t-2)" "'
                            4 `" "Pre-treatment" "citation (t-3)" "' 
                            5 `" "Pre-treatment" "citation (t-4)" "' 
                            1 `" "Patent age" "(in quarter)" "' 
                            6 "# of claims"
                        )
                       
                    )
                    yline(0.2, lpattern(dash) lcolor("$c_gray") noextend)
                    ytitle("|SMD|") 
                    legend( order(1 `"{it:{&lambda} = 0}"' 2 "`opt_lam_text'"  3 `"{it:{&lambda} = 1}"' - ""  4 `"Before matching"' )  col(3) row(2) textwidth(13))
                    legend(region(lcolor(white)))
                    marker(1, `m_lambda0')
                    marker(2, `m_lambda_opt')
                    marker(3, `m_lambda1')
                    marker(4, `m_before')
                    graphregion(color(white)) 
                    graphregion(margin(medium))
                    plotregion(margin(small))

                ;
                #delimit cr

                graph export "${out}\01 Comparison of SMDs - `sam'.png", replace width(3500) height(2000)


            restore
        
    } */

* ==============================================================================
* C. Load and clean and plot for PC's 
* ==============================================================================
    ** Load
    u "G:\My Drive\uc3m PhD\05 Analysis\01 Main\01 Stata\01 Main\04 Matching quality\out\02. SMDs.dta", clear

    ** Filter and clean
    keep if regexm(cov, "pc")
    keep if inlist(lambda, 0, 1, 99) | (lambda == 0.6 & acq_type == "M&A") | (lambda == 0.7 & acq_type == "Off deal")
    tostring lam, replace
    replace lam = subinstr(lam, ".", "0", .)
    replace smd = abs(smd)

    ** Setup
    egen id = group(acq bl), label
    reshape wide smd, i(id covariate) j(lam) str

    gen smd_opt = smd06
    replace smd_opt = smd07 if mi(smd06)
    assert !mi(smd_opt)

    ** Plot
    graph set window fontface "Arial Unicode MS"
    graph set print  fontface "Arial Unicode MS"
    local isin = uchar(8712)
    #delimit ;
        graph hbox  smd0  smd_opt smd1 smd99, 
        over(
            id ,
            relabel(
                1 `" "M&A" "baseline" "'
                2 `" "M&A" "top-tech" "'
                3 `" "Off deal" "baseline" "'
                4 `" "Off deal" "top-tech" "' 
            )
            gap(*1.8)
            
        )
        yline(0.2, lpattern(dash) lcolor("$c_gray") noextend)
        ytitle("|SMD|") 
        legend( order(1 `"{it:{&lambda} = 0}"' 2 `"{it:{&lambda} = 0.6/0.7}"'  3 `"{it:{&lambda} = 1}"' - ""  4 `"Before matching"' )  col(3) row(2) textwidth(17))
        legend(region(lcolor(white)) symxsize(*0.5) size(small))
        box(1, color($c_blue)) marker(1, mcolor($c_blue))
        box(2, color($c_red)) marker(2, mcolor($c_red))
        box(3, color($c_pink)) marker(3, mcolor($c_pink))
        box(4, color($c_gray)) marker(4, mcolor($c_gray))
        ylabel(, nogrid)
        graphregion(color(white)) 
        graphregion(margin(medium))
        plotregion(margin(small))

    ;
    #delimit cr
    graph export "${out}\01 Comparison of SMDs - PCs by sample.png", replace width(3500) height(2000) 

    "Cambria Math"



    