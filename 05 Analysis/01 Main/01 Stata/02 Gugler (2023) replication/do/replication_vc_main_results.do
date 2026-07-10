********************************************************************************
* Code to replicate main VC results of Gugler, Scuezs, Wohak (2025)
* Set current working directory to replication_package/
********************************************************************************

* Change working directory
// cd ../replication_package/
// ssc install eventstudyinteract
use data/estimation_sample_vc.dta, clear

* Figure 5 (right)
eventstudyinteract ln_money early t_5-t_2 t0 t1 - t20, absorb(id year) cohort(cohort) control_cohort(c) vce(cluster id)
matrix C = e(b_iw)
mata st_matrix("A",sqrt(diagonal(st_matrix("e(V_iw)"))))
matrix C = C \ A'
matrix list C
coefplot matrix(C[1]), se(C[2]) vertical keep(t_5 t_4 t_3 t_2 t0 t1 t2 t3 t4 t5) rename(t_5 = "t=-5" t_4 = "t=-4" t_3 = "t=-3" t_2 = "t=-2" t0="t=0" t1 = "t=1" t2 = "t=2" t3 = "t=3" t4 = "t=4" t5 = "t=5") xline(4.5) yline(0) name(pretrends_inv, replace)

* Table 2 Cols 1-3
eventstudyinteract ln_money treatpost, absorb(id year) cohort(cohort) control_cohort(c) vce(cluster id)
eventstudyinteract ln_money treatpost????, absorb(id year) cohort(cohort) control_cohort(c) vce(cluster id)
eventstudyinteract ln_money treatpost??, absorb(id year) cohort(cohort) control_cohort(c) vce(cluster id)

* Table 4 VC investment
eventstudyinteract ln_money treatpost??????, absorb(id year) cohort(cohort) control_cohort(c) vce(cluster id)

* Table 5 Col 1
reghdfe ln_money 1.treated#1.post 1.treated#1.post#c.tech_overlap_max 1.treated#1.post#c.product_overlap_max, ///
absorb(id year) keepsingletons cl(id)

* Table 5 Col 2
reghdfe ln_money 1.treated#1.post#1.treat_* 1.treated#1.post#1.treat_*#c.tech_overlap_max ///
1.treated#1.post#1.treat_*#c.product_overlap_max, absorb(id year) keepsingletons cl(id)

* Table 7 Col VC Investment
gen young = (age_at_acq < 11)
sum ln_money, de
reghdfe ln_money 1.treated#1.post 1.treated#1.post#1.(bin_common private after2010 morethanfiveyears) ///
	, absorb(id year) keepsingletons cl(id)

* EOF