
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
        gl dta "${desc}/dta"
        gl out "${desc}/out"
        gl raw "${desc}/raw"
        gl temp "${desc}/temp"

********************************************************************************
* B. Load grid results for citation
********************************************************************************
    ** Load CSV
    import delimited "G:\My Drive\PhD Data\11 Matches\optimization results\citation_no_exact_match_on_grantyear\grid_results_log.csv", clear
awd
    ** Round lambdas and drop duplicates
    replace lambda = round(lambda, 0.01)
    duplicates drop

    ** Generate sample variable
    replace top_tech = "Top-Tech" if top_tech == "True"
    replace top_tech = "Baseline" if top_tech == "False"
    
    tostring top_tech_threshold, replace
    replace top_tech_threshold = "" if top_tech == "Baseline"

    gen sample = acq_type + " - " + top_tech if top_tech == "Baseline"
    replace sample = acq_type + " - " + top_tech + " - p" + top_tech_threshold if top_tech == "Top-Tech"
    assert !mi(sample)

    ** Place periods
    replace baseline_begin_period = round((baseline_begin_period - 1) / 2)

    rename number_of_matches K
    ** Label
    label var acq_type "Acquisition type"
    label var top_tech "Tech. class"
    label var top_tech_threshold "Top-Tech threshold"
    label var sample "Sample"
    label var caliper "Caliper (threshold for scaled hybrid distance)"
    label var baseline_begin_period "Total # placebo periods"
    label var K "Number of matches (K)"
    label var lambda "Lambda"
    label var mse_diff "MSE"
    label var total_num_patents "Total # patents"
    label var num_dropped "Dropped # patents"

********************************************************************************
* C. Mark the optimal lambda that minimizes MSE
********************************************************************************
    ** Sort and mark
    bys sample K caliper baseline_begin_period (mse_diff): gen optimal = _n == 1
    gen x = lam if optimal == 1
    gen y = mse_diff if optimal == 1
    bys sample K caliper baseline_begin_period (x): replace x = x[1] if mi(x)
    bys sample K caliper baseline_begin_period (y): replace y = y[1] if mi(y)

    label var y "Optimal lambda"


********************************************************************************
* D. Prepare graphs
********************************************************************************
    ** String caliper
    gen aux = caliper 
    replace aux = aux*100
    tostring aux, replace
    replace aux = aux + "%" 


    ** Locals
    levelsof sample, local(samples)
    levelsof K, local(Ks)
    levelsof aux, local(calipers)
    levelsof baseline_begin_period, local(periods)

    foreach period of local periods{
        foreach caliper of local calipers{
            foreach K of local Ks{
                    ** Subset
                    preserve
                        keep if K == `K' & aux == "`caliper'" & baseline_begin_period == `period'

                        ** Graph
                        #delimit ;
                            sepscatter mse_diff lambda, separate(sample) recast(connect)  sort 
                            addplot(
                                scatter y x, mcolor(red) msymbol(circle_hollow) msize(large) 
                                title(" # Matches = `K', Caliper = `caliper', Periods = `period'") 
                                xtitle("Lambda") 
                                ytitle("MSE") 
                                legend(pos(2) ring(0) cols(2) size(vsmall))
                                graphregion(color(white))
                                bgcolor(white) 
                            )
                            ;
                        #delimit cr
                        ** Save
                        ** Actual caliper
                        replace aux = subinstr(aux, ".", ",", .)
                        local caliper_helper = aux[1]

                        tab caliper

                        graph export "${out}/`period' periods/MSE vs. Lambda - `K' matches_Caliper `caliper_helper'.png", replace width(4000) height(3000)

                    restore
                }
            }
        }
    




