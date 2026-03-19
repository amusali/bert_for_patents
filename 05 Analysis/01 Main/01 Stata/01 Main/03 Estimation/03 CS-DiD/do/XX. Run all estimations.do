* ==============================================================================
* A. Set paths
* ==============================================================================
    
    clear all
    capture log close
    set more off
    set type double, perm
    set excelxlsxlargefile on

    ** set path
    gl do  "C:\Users\aliwk\OneDrive\Desktop\PhD (repos)\BERT\bert_for_patents\05 Analysis\01 Main\01 Stata\01 Main\03 Estimation\03 CS-DiD\do"

* ==============================================================================
* B. Run 
* ==============================================================================

    ** Off deal 
    do "${do}\11. CSDID - Off deal.do"

    ** M&A 
    do "${do}\10. CSDID - M&A.do"