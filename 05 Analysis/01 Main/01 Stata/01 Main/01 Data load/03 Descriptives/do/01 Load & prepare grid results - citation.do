
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
    ** Load CSV
    import delimited "G:\My Drive\PhD Data\11 Matches\optimization results\paper\grid_results_log.csv", clear

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
    label var lambda "λ"
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
    levelsof acq_type, local(acq_types)
    levelsof K, local(Ks)
    levelsof aux, local(calipers)
    levelsof baseline_begin_period, local(periods)
    levelsof top_tech, local(techs)
    local i = 1
    foreach period of local periods{
        foreach tech of local techs{
                    ** Subset
                    preserve
                        keep if top_tech == "`tech'" & baseline_begin_period == `period'

                        di in red "top_tech == `tech' & baseline_begin_period == `period'"

                        if `i' == 5{
                            local b1title = "Baseline"
                        }
                        else if `i' == 6{
                            local b1title = "Top-Tech"
                        }
                        else{
                            local b1title = ""
                        }

                        if `i' == 1{
                            local ytitle = "{it:n}{sup:{it:p}} {it:=} {it:4}"
                        }
                        else if `i' == 3{
                            local ytitle = "{it:n}{sup:{it:p}} {it:=} {it:6}"
                        }
                        else if `i' == 5{
                            local ytitle = "{it:n}{sup:{it:p}} {it:=} {it:8}"
                        }
                        else{
                            local ytitle = ""
                        } 
                        ** Graph
                        #delimit ;
                            sepscatter mse_diff lambda, separate(acq_type) recast(connect)  sort lc("$c_blue" "$c_red") mc("$c_blue" "$c_red") 
                            addplot(
                                scatter y x, mcolor("$c_green") msymbol(circle_hollow) msize(large) 
                                xtitle("{&lambda}") 
                                ytitle("MSE") 
                                graphregion(color(white))
                                bgcolor(white) 
                                legend(order(1 2) position(1) col(2) ring(0))
                            )
                            b2title("`b1title'"  )
                            l2title("`ytitle'" )
                            ylab(, grid glwidth(thin) glpattern(solid)) 
                            xlabel(, nogrid)
                            name(g`i', replace)
                            ;
                        #delimit cr

                        *graph save g`i'.gph, replace

                        local i = `i' + 1
                        ** Save
                        /* ** Actual caliper
                        replace aux = subinstr(aux, ".", ",", .)
                        local caliper_helper = aux[1] */

                        graph export "${out}/`period' periods/Paper - MSE vs. Lambda - `tech'.png", replace width(4000) height(3000)

                    restore
                }
            }

    ** Comine graphs and save
    grc1leg2 g1 g2 g3 g4 g5 g6, cols(2) iscale(0.5) xcommon position(1) ring(0)  ytol xtob dots ytsize(vsmall) xtsize(vsmall) xsize(4.5) ysize(4) 
    graph export "${out}/Paper - MSE vs. Lambda - Combined.png", replace width(4000) height(3000)
    
     /*/
    ** Combine 6 graphs into 1
    #delimit ;
        graph combine g1 g2 g3 g4 g5 g6,
        cols(2) iscale(0.5) 
        xcommon

        graphregion(margin(b=18)) 
        
    ;
    #delimit cr

        
       
    




