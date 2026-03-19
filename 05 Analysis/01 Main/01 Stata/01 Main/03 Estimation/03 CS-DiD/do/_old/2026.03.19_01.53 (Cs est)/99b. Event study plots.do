* ==============================================================================
* A. Set paths
* ==============================================================================
    clear all
    capture log close
    set more off
    set type double, perm
    set excelxlsxlargefile on

    ** set path
    gl analysis "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main"
    gl python "${analysis}/00 Python data"
    gl stata "${analysis}/01 Stata"
    
    gl est "${stata}/01 Main/03 Estimation"
    gl csdid "${est}/03 CS-DiD"
        gl aux "${csdid}/_aux"
        gl do  "${csdid}/do"
        gl dta "${csdid}/dta"
        gl out "${csdid}/out/graphs/paper"
        gl raw "${csdid}/raw"
        gl temp "${csdid}/temp"
        gl log "${csdid}/log"


    ** Start log
    log using "${log}/99b. Event study plots.log", replace
    timer clear 1
    timer on 1

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
* B. Load estimates and parse results
********************************************************************************
    ** Load estimates 
    u "${dta}\CSDID Estimates Summary (extended).dta", clear 

    ** Filter here 
    local pre_treatment_period = 4
    local caliper = 0.05

    keep if pretreatment_periods == `pre_treatment_period' & caliper == `caliper'
    keep if inlist(lambda, 0, 1) | (lambda == 0.6 & acq_type == "M&A") | (lambda == 0.7 & acq_type == "Off deal")

    assert _N == 4 * 3  // 4 configurations and 3 lambdas
    sort acq_type bl_tt lambda

    gen sample = acq_type + ", " + bl_tt + ", " + string(lambda)

    ** Loop over the configurations and plot
    qui count
    forvalue i = 1/12{
        preserve
            ** Filter
            keep if _n == `i'
            dropmiss, force

            ** Crazy Renaming 
            rename es* *
            rename _#* _p#*
            rename _* *
            rename *_p pval*
            rename *_se se*
            rename m* point_estm*
            rename p# point_estp*

            ** Reshape
            reshape long point_est se pval, i(t_config) j(period) str
            drop if period == "simple"
            replace period = subinstr(period, "m", "-", .)
            replace period = subinstr(period, "p", "", .)
            destring period, replace
            sort period

            ** Bounds for 95% CI
            gen ci_lower = point_est - 1.96 * se
            gen ci_upper = point_est + 1.96 * se

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
                (rcap ci_lower ci_upper period if period < 0, lcolor("$c_blue") lwidth(0.5))
                (rcap ci_lower ci_upper period if period >= 0, lcolor("$c_red") lwidth(0.5))
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
    } 

********************************************************************************
* C. Combine plots for each sample
********************************************************************************

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
