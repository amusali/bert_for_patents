* ==============================================================================
* A. Set paths
* ==============================================================================
    clear all
    capture log close
    set more off
    set type double, perm
    set excelxlsxlargefile on

    ** set path
    gl analysis "G:\My Drive\uc3m PhD\05 Analysis\01 Main"
    gl python "${analysis}/00 Python data"
    gl stata "${analysis}/01 Stata"
    
    gl est "${stata}/01 Main/03 Estimation"
    gl csdid "${est}/03 CS-DiD"
        gl aux "${csdid}/_aux"
        gl do  "${csdid}/do"
        gl dta "${csdid}/dta"
        gl out "${csdid}/out/graphs/paper/nearby patents"
        gl raw "${csdid}/raw"
        gl temp "${csdid}/temp"
        gl log "${csdid}/log"


    ** Start log
   * log using "${log}/99b. Event study plots.log", replace
    timer clear 1
    timer on 1

    ** Colours - source: https://lospec.com/palette-list/new-tableau-10
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

    /* global c_blue    "85 124 167"
    global c_orange  "233 138 46"
    global c_red     "213 82 90"
    global c_teal    "128 185 178"
    global c_green   "102 161 79"

    global c_yellow  "233 199 76"
    global c_purple  "170 122 160"
    global c_pink    "247 155 168"
    global c_brown   "152 116 94"
    global c_gray    "185 176 171" */

********************************************************************************
* B. Load estimates and parse results
********************************************************************************
    ** Load estimates 
    u "${dta}\CSDID Estimates Summary - Nearby Patents - (COLAB, Cross Section estimates).dta", clear 

    ** Filter here 
    local pre_treatment_periods  4 6 8
    local calipers 0.05 0.10

    ** Sample
    gen sample = acq_type + ", " + bl_tt + ", " + string(pretreatment_periods) + "q, caliper" + string(caliper) + ", " +string(lambda) + ", " + subinstr(outcome, "log_nearby_patents_", "", .)

        
    ** Loop over the configurations and plot

    ** Filter
    *assert _N == 4 * 3  // 4 configurations and 3 lambdas
    sort acq_type bl_tt lambda
    count
    /* forvalue i = 1/139{
        di in red "AWDWAD"
        preserve
            ** Filter
            keep if _n == `i'
            dropmiss, force

            ** Crazy Renaming 
            rename es* *
            rename _#* _p#*
            rename _* *
            *rename *_p pval*
            rename *_se se*
            rename *_cl cl*
            rename *_cu cu*
            rename m* point_estm*
            rename p# point_estp*
            

            ** Reshape
            reshape long point_est se cl cu, i(t_config) j(period) str
            drop if period == "simple"
            replace period = subinstr(period, "m", "-", .)
            replace period = subinstr(period, "p", "", .)
            destring period, replace
            sort period

            ** Plot 
            qui sum period, detail
            local min_period = r(min)
            local max_period = r(max)
            local sample  = sample[1]
            local sample = subinstr("`sample'", ".", "0", .)
            
            if lambda[1] == 0 {
                local title `"{it:{&lambda} = 0}"'
            }
            else if lambda[1] == 0.6 {
                local title `"{it:{&lambda} = 0.6}"'
            }
            else if lambda[1] == 0.7 {
                local title `"{it:{&lambda} = 0.7}"'
            }
            else if lambda[1] == 1 {
                local title `"{it:{&lambda} = 1}"'
            }

            #delimit ;
                graph twoway 
                (rcap cl cu period if period < 0, lcolor("$c_blue") lwidth(0.5))
                (rcap cl cu period if period >= 0, lcolor("$c_red") lwidth(0.5))
                (scatter point_est period if period < 0, mcolor("$c_blue") msymbol(circle) msize(medium))
                (scatter point_est period if period >= 0, mcolor("$c_red") msymbol(circle) msize(medium)),
                yline(0, lpattern(dash) lcolor("$c_pink")) 
                xline(0, lpattern(dash) lcolor("$c_gray") noextend)
                xlabel(`min_period'(1)`max_period', nogrid) 
                title("`title'", size(large))
                ytitle("ATT") 
                xtitle("Quarters since Treatment") 
                legend(order(1 "Pre-treatment" 2 "Post-treatment") position(1) col(2) ring(0))
                graphregion(color(white)) 

                saving("${out}\Paper - Event Study - `sample'.gph", replace)
            ;
            #delimit cr

            graph export "${out}\Paper - Event Study - `sample'.png", replace width(3000) height(2000)

        restore
    }   */



********************************************************************************
* C. Combine plots for each sample
********************************************************************************
    ** Loop over samples
    replace sample = subinstr(sample, ", 0", "", .)
    replace sample = subinstr(sample, ", .6", "", .)
    replace sample = subinstr(sample, ", .7", "", .)
    replace sample = subinstr(sample, ", 1", "", .)

    replace sample = subinstr(sample, "caliper.05", "caliper005", .)
    replace sample = subinstr(sample, "caliper.1", "caliper01", .)

    local outcome_types = `" " all_d150" " all_d200" " gafam_d150" " gafam_d200"  "'

    replace sample = subinstr(sample,  "  all_d100", "", .)
    replace sample = subinstr(sample,  "  gafam_d100", "", .)

    foreach outcome of local outcome_types{
        replace sample = subinstr(sample,  "`outcome'", "", .)
        replace sample = strtrim(sample)
        replace sample = subinstr(sample,  "caliper005,", "caliper005", .)
    }

    qui levelsof sample, local(samples)

    foreach sam of local samples{
        if regexm("`sam'", "8q") continue
        foreach outcome of local outcome_types{
            if regexm("`sam'", "M&A"){
                local optimal_lam  = "06"
            }
            else{
                local optimal_lam  = "07"
            }

            #delimit ;
                grc1leg2 
                "${out}\Paper - Event Study - `sam', 0, `outcome' .gph"  
                "${out}\Paper - Event Study - `sam', `optimal_lam', `outcome' .gph"     
                "${out}\Paper - Event Study - `sam', 1, `outcome' .gph", 
                ycommon col(3) ytol xtob xsize(9) ysize(4)
            ;
            #delimit cr

            
            graph export "${out}\Paper - Event Study - `sam' - `outcome' - all lambdas.png", replace width(4500) height(2000)

            }
    }
        
/*
    *---------------------------------------------------------*
    * C.1. M&A - baseline
    *---------------------------------------------------------*

        #delimit ;
            grc1leg2 
            "${out}\Paper - Event Study - M&A,  baseline, 0.gph"  
            "${out}\Paper - Event Study - M&A,  baseline, 06.gph"     
            "${out}\Paper - Event Study - M&A,  baseline, 1.gph", 
            ycommon col(3) ytol xtob xsize(9) ysize(4)
        ;
        #delimit cr

        
        graph export "${out}\Paper - Event Study - M&A baseline - all lambdas.png", replace width(4500) height(2000)

    *---------------------------------------------------------*
    * C.2. M&A - top-tech
    *---------------------------------------------------------*
        #delimit ;
            grc1leg2 
            "${out}\Paper - Event Study - M&A,  top-tech, 0.gph"  
            "${out}\Paper - Event Study - M&A,  top-tech, 06.gph"     
            "${out}\Paper - Event Study - M&A,  top-tech, 1.gph", 
            ycommon col(3) ytol xtob xsize(9) ysize(4)
        ;
        #delimit cr
        
        graph export "${out}\Paper - Event Study - M&A top-tech - all lambdas.png", replace width(4500) height(2000)

    *---------------------------------------------------------*
    * C.3. Off deal - baseline
    *---------------------------------------------------------*
        #delimit ;
            grc1leg2 
            "${out}\Paper - Event Study - Off deal,  baseline, 0.gph"  
            "${out}\Paper - Event Study - Off deal,  baseline, 07.gph"     
            "${out}\Paper - Event Study - Off deal,  baseline, 1.gph", 
            ycommon col(3) ytol xtob xsize(9) ysize(4)
        ;
        #delimit cr
        
        graph export "${out}\Paper - Event Study - Off deal baseline - all lambdas.png", replace width(4500) height(2000)

    *---------------------------------------------------------*
    * C.4. Off deal - top-tech
    *---------------------------------------------------------*
        #delimit ;
            grc1leg2 
            "${out}\Paper - Event Study - Off deal,  top-tech, 0.gph"  
            "${out}\Paper - Event Study - Off deal,  top-tech, 07.gph"     
            "${out}\Paper - Event Study - Off deal,  top-tech, 1.gph", 
            ycommon col(3) ytol xtob xsize(9) ysize(4)
        ;
        #delimit cr
        
        graph export "${out}\Paper - Event Study - Off deal top-tech - all lambdas.png", replace width(4500) height(2000) 
