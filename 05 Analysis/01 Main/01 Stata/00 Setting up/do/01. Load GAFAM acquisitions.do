* Set path
gl stata "C:\Users\amusali\Desktop\uc3m PhD\05 Analysis\01 Main\01 Stata\"

** Import data
import excel "${stata}\raw\sdc_gafam\SDC Platinum sdc_gafam_usonly_completed.xlsx", sheet("Request 7") cellrange(A3:AG728) firstrow

** Clean
replace FirmValueUSDMillions = subinstr( FirmValueUSDMillions, ",", "", .)
destring FirmValueUSDMillions, replace 
