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
        gl out "${csdid}/out/graphs/paper"
        gl raw "${csdid}/raw"
        gl temp "${csdid}/temp"
        gl log "${csdid}/log"


    ** Start log
   * log using "${log}/99b. Event study plots.log", replace
    timer clear 1
    timer on 1


* ==============================================================================
* B. M&A Leave-one-out of top 5 M&A deals
* ==============================================================================

    ** Load 
    u "${dta}\CSDID Estimates Summary (COLAB, Cross Section estimates) - nemogy with weights.dta", clear

    ** FIlter and reshape 
    keep if regexm(est_con, "macloo")
    keep est_config acq_type bl_tt tt_threshold pretreatment_periods posttreatment_periods caliper lambda est_simple
    replace est_config = subinstr(est_config, "nemogy_with_weigths_macloo", "", .)
    destring est_config, replace
    reshape wide est_simple, i(lam) j(est_config)
    order lam est_simple999 est_simple136 est_simple288 est_simple183 est_simple260 est_simple278

    save "${dta}\99 Robustness - M&A leave one out estimates.dta", replace


* ==============================================================================
* C. Off deal leave-one-out 
* ==============================================================================
    ** Load 
    u "${dta}\CSDID Estimates Summary (COLAB, Cross Section estimates) - nemogy with weights.dta", clear

    ** FIlter and reshape 
    keep if regexm(est_con, "odloo")
    keep est_config acq_type bl_tt tt_threshold pretreatment_periods posttreatment_periods caliper lambda est_simple
    replace est_config = subinstr(est_config, "nemogy_with_weigths_odloo", "", .)
    destring est_config, replace
    reshape wide est_simple, i(lam) j(est_config)

    save "${dta}\99 Robustness - Off deal leave one out estimates.dta", replace


* ==============================================================================
* D. Drop problematic cohorts in M&A samples
* ==============================================================================

    ** Load
    u "${dta}\CSDID Estimates Summary (COLAB, Cross Section estimates) - nemogy with weights.dta", clear

    ** Filter
    keep if inlist(est_config, "nemogy_with_weigths_cc", "nemogy_with_weigths") & caliper == 0.05 & acq_type == "M&A" & pretreatment_periods == 4 
    drop if contains_pcs == 1 & lam == 1
    assert _N == 12
    drop es_*

    ** Reshape
    replace est_con = "cc" if est_con == "nemogy_with_weigths_cc"
    replace est_con = "nocc" if est_con == "nemogy_with_weigths"

    reshape wide pre_trend_all_pval pre_trend_walt_statistic est_simple, i( acq_type bl_tt lambda) j(est_config) str

    order acq_type bl_tt lambda est_simplenocc est_simplecc pre_trend_all_pvalnocc pre_trend_all_pvalcc  

    rename *_all_* **
    rename est_simple* es_*
    rename pre_trendpval* pval_*



