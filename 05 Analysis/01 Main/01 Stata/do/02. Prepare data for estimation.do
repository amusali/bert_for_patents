** Set Path
gl stata "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main\01 Stata"
gl raw  "$stata\raw"

** Load data
import excel "$stata\patents_to_quarterly_df.xlsx", sheet("Sheet1") firstrow

** Drop the index column
drop A

** Destring
foreach var of varlist * {
    destring `var', replace
}

** Drop duplicates -- temp solution
egen id = group( matched_pair_id patent_id)
duplicates tag id quarters_to_acquired_date, gen(dup)
drop if dup

gisid id quarters_to_acquired_date

** XT setup
xtset id quarters_to_acquired_date

** Get year
gen year_of_quarter = yofd(dofc(quarter_date))
gen year_of_acquisition = yofd(dofc(acquired_date))

** Shift the qurater to date variable
replace quarters_to_acquired_date = quarters_to_acquired_date + 16
