
	****************************************************************************
	* SDC mergers
	****************************************************************************
	
import excel using data/techmergers.xlsx, clear first cellrange(A2)

keep if nicht==.

rename DateAnn date
gen year=year(date)
replace year=yearadd if year==.
drop yearadd
keep if AcquirorNation=="United States" | AcquirorNation==""

ren AcquirorName aname
order date year aname
so aname date

* 1991, different alphabet
drop if aname=="Alphabet Technologies Inc"
* food
drop if aname=="Apple South Inc"

replace AcquirorUltimateParent="ALPHABET" if regexm(AcquirorUltimateParent, "Alphabet")	
replace AcquirorUltimateParent="APPLE" if regexm(AcquirorUltimateParent, "Apple")	
replace AcquirorUltimateParent="AMAZON" if regexm(AcquirorUltimateParent, "Amazon")	
replace AcquirorUltimateParent="FACEBOOK" if regexm(AcquirorUltimateParent, "Facebook")	
replace AcquirorUltimateParent="MICROSOFT" if regexm(AcquirorUltimateParent, "Microsoft")	
		
keep if Status=="Completed" | Status==""

drop RankDate DateEffective DateEffectiveUnconditional DateWithdrawn TargetCompanyDateofFin Status
drop TargetShortBusinessDescripti TargetIndustrySector HighTechIndustry TargetState AcquirorShortBusinessDescript AcquirorIndustrySector R AcquirorState Synopsis HistoryFileEvent EnterpriseValuemil EquityValuemil TargetNetSalesLTMmil EBITLastTwelveMonthsMil NetIncomeLastTwelveMonths TargetNetAssetsmil TargetTotalAssetsmil TargetEBITDALTMmil TargetBookValuePerShareLTM ValueofTransactionmil nicht1repurchaseauchnich AM AN AO AP AQ AR

gen tname=TargetName

duplicates drop

foreach s in Corp Inc S.R.O. Ltd LLC Llc SL Oy BV AG GmbH Srl ASA AB SAS SA & A/S " ." ' {
	replace tname=subinstr(tname, "`s'", "", .)
	}
replace tname=trim(tname)
replace tname=upper(tname)
replace tname=stritrim(tname)

* make tname unique:
drop if aname=="Claris Corp(Apple Computer)" & date==10419 & tname=="STYLEWARE"
* wikipedia: amazon bought audible in 2008, drop others
drop if tname=="AUDIBLE" & year!=2008
* wikipedia: duolingo received 45m from google in 2015
drop if tname=="DUOLINGO" & year!=2015
* one deal in 1997, one in 99, keep earlier
drop if tname=="LERNOUT HAUSPIE SPEECH" & year!=1997
* similar
drop if tname=="WEBTV NETWORKS" & year!=1996

* these are dups (with previously existing deals) in the add deals from the data appendix of the comp. report
drop if tname=="BELUA"
drop if tname=="CANVASTECHNOLOGY"
drop if tname=="CRTLLABS"
drop if tname=="DMARKBROADCASTING"
drop if tname=="DRIVYSHOT"
drop if tname=="GLOWALLA"
drop if tname=="HTCSMARTPHONEDIVISION"
drop if tname=="INTERNETMOVIEDATABASEIMDB"
drop if tname=="JETPACK"
drop if tname=="LITTLEEYELABS"
drop if tname=="POLARROSE"
drop if tname=="PROOGEO"
drop if tname=="SAFABATRANSLATIONSYSTEMS"
drop if tname=="THINKBOXSYSTEMS"
drop if tname=="THREASY"
drop if tname=="URCHINSOFTWAREORATION"
drop if tname=="WIFISLAM"

* change names of manual patstat matches
preserve // add manual matches to patstat
	import excel using data/targets_nopats.xls, clear first
	ren B tname
	keep tname name
	drop if name==""
	tempfile add
	save `add'
restore

merge 1:1 tname using `add', nogen
replace tname=name if name!=""
drop name

save data\deals, replace

	****************************************************************************
	* download GAFAM patents
	****************************************************************************

odbc query "patstat_spring_2020", dialog(complete)

* generate IPC/year panel for acquirers

preserve
	foreach name in "AMAZON TECHNOLOGIES" GOOGLE AMAZON.COM FACEBOOK "FACEBOOK TECHNOLOGIES" APPLE "MICROSOFT CORPORATION" "MICROSOFT TECHNOLOGY LICENSING" {
		disp "`name'"
		odbc load, exec("SELECT tls206_person.psn_name, tls207_pers_appln.appln_id, tls201_appln.earliest_filing_year, tls201_appln.appln_kind,tls201_appln.granted , tls201_appln.nb_citing_docdb_fam, tls201_appln.nb_applicants, tls201_appln.nb_inventors, tls209_appln_ipc.ipc_class_symbol FROM tls206_person  INNER JOIN tls207_pers_appln ON tls206_person.person_id=tls207_pers_appln.person_id  INNER JOIN tls201_appln ON tls201_appln.appln_id=tls207_pers_appln.appln_id INNER JOIN tls209_appln_ipc ON tls209_appln_ipc.appln_id=tls207_pers_appln.appln_id WHERE (tls206_person.psn_name LIKE '`name'') AND tls201_appln.appln_kind = 'A' AND tls201_appln.granted = 'Y'  ") clear
		loc name2 = subinstr("`name'", " ", "",.)
		save data/`name2', replace
		}
restore		
	
* loadsql ado
	cap drop program loadsql
	program define loadsql
	*! Load the output of an SQL file into Stata, version 1.3 (dvmaster@gmail.com)
	*! Based on http://stackoverflow.com/a/12272627/173351
	version 14.1
	syntax using/, DSN(string) [User(string) Password(string) CLEAR NOQuote LOWercase SQLshow ALLSTRing DATESTRing]
	#delimit;
	tempname mysqlfile exec line;
	!cat "`using'" | ruby -e "print ARGF.read.gsub(/^\s+--.*$/, '').gsub(/\s+/, ' ')" > "`using'_oneline";
	file open `mysqlfile' using `"`using'"', read text;
	file read `mysqlfile' `line';
	while r(eof)==0 {;
		local `exec' `"``exec'' ``line''"';
		file read `mysqlfile' `line';
	};
	file close `mysqlfile';
	!rm "`using'_oneline";
	odbc load, exec(`"``exec''"') dsn(`"`dsn'"') user(`"`user'"') password(`"`password'"') `clear' `noquote' `lowercase' `sqlshow' `allstring' `datestring';
	end;

	
	
	****************************************************************************
	* generate IPC/year panel for targets
	****************************************************************************

use data\deals, clear	

levelsof tname, local(names)
foreach name of local names {
	loc query " "`query'" "" (tls206_person.psn_name LIKE '`name'') OR ""
	}
di `query'
 file open sql using data/sql3.txt, write
 file write sql `query'

* req.sql has been manually edited from the above macro containing all target names
clear
loadsql using "data/req.sql", dsn("patstat_spring_2020")		

ren earliest_filing_year patentyear
drop if length(ipc_class_symbol)==4 
replace ipc_class_symbol = subinstr(ipc_class_symbol, " ", "", .)
replace ipc_class_symbol = subinstr(ipc_class_symbol, "/", "", .)

drop appln_kind granted nb_inventors nb_applicants
duplicates drop

bysort psn_name ipc patentyear: gen patents=_N
bysort psn_name ipc patentyear: egen patentcites=total(nb_citing_docdb_fam)
drop appln_id nb_citing_docdb_fam
duplicates drop

egen id = group(psn_name ipc)
xtset id patentyear

compress
ren psn_name tname
recast str50 tname
merge m:1 tname using deals, keep(3) keepusing(AcquirorUltimateParent year) nogen
ren year year_acquired

compress
save data\target_patents, replace

	****************************************************************************
	* generate IPC/year panel for GAFAM
	****************************************************************************

clear

	foreach name in AMAZONTECHNOLOGIES GOOGLE AMAZON.COM FACEBOOK APPLE MICROSOFTCORPORATION MICROSOFTTECHNOLOGYLICENSING {
		
		import delimited using data/patents/`name', clear
		*loc name2 = subinstr("`name'", " ", "",.)
		save data/patents/`name', replace
		}

clear
     
append using data/patents\AMAZONTECHNOLOGIES
append using data/patents\GOOGLE
append using data/patents\AMAZON.COM
append using data/patents\FACEBOOK
append using data/patents\APPLE
append using data/patents\MICROSOFTCORPORATION
append using data/patents\MICROSOFTTECHNOLOGYLICENSING
drop v*

ren ïpsn_name AcquirorUltimateParent
replace AcquirorUltimateParent="ALPHABET" if regexm(AcquirorUltimateParent, "GOOGLE")	
replace AcquirorUltimateParent="AMAZON" if regexm(AcquirorUltimateParent, "AMAZON")	
replace AcquirorUltimateParent="MICROSOFT" if regexm(AcquirorUltimateParent, "MICROSOFT")	
tab AcquirorUltimateParent

codebook appln_id if AcquirorUltimateParent=="AMAZON"
codebook appln_id if AcquirorUltimateParent=="APPLE"
codebook appln_id if AcquirorUltimateParent=="FACEBOOK"
codebook appln_id if AcquirorUltimateParent=="ALPHABET"
codebook appln_id if AcquirorUltimateParent=="MICROSOFT"

ren earliest_filing_year patentyear
*gen str4 ipc4=ipc_class_symbol

* retain ipc8 level!
drop if length(ipc_class_symbol)==4 
replace ipc_class_symbol = subinstr(ipc_class_symbol, " ", "", .)
replace ipc_class_symbol = subinstr(ipc_class_symbol, "/", "", .)
drop appln_kind granted nb_inventors nb_applicants
duplicates drop //patent level ipc8 

bysort AcquirorUltimateParent ipc patentyear: gen patentsacq=_N
bysort AcquirorUltimateParent ipc patentyear: egen patentcitesacq=total(nb_citing_docdb_fam)
drop appln_id nb_citing_docdb_fam
duplicates drop // patent level ipc -> firm/ipc level

egen id = group(AcquirorUltimateParent ipc)
xtset id patentyear
drop if patentyear<1990 | patentyear>2021
tsfill, full
bysort id (AcquirorUltimateParent): replace AcquirorUltimateParent=AcquirorUltimateParent[_N]
bysort id (ipc): replace ipc=ipc[_N]
recode patent* (.=0)

compress
merge 1:m AcquirorUltimateParent patentyear ipc_class_symbol using data/target_patents, keep(1 3)

* _m==1: firm/ipc/years of acquirers in which no target had patents
* _m==2: patents of target in tech class, in which acq has no patents 
* _m==3: both acq and tar have patents
* m2/(m2+m2)=.16 --> 84% of target patents in overlapping IPCs

gen p=(patentyear>year_acquired & year_acquired!=.)
bysort AcquirorUltimateParent ipc patentyear: egen post=max(p)
bysort AcquirorUltimateParent ipc: egen treated=max(post)
bysort AcquirorUltimateParent ipc (patentyear): replace post=1 if post[_n-1] & _n!=1

bysort AcquirorUltimateParent ipc patentyear: egen targetallpat=total(patents)
bysort AcquirorUltimateParent ipc patentyear: egen targetallcit=total(patentcites)

bysort AcquirorUltimateParent ipc patentyear: egen tpp=total(patents) if p
bysort AcquirorUltimateParent ipc patentyear: egen tpc=total(patentcites) if p
bysort AcquirorUltimateParent ipc patentyear: egen targetpostpat=max(tpp)
bysort AcquirorUltimateParent ipc patentyear: egen targetpostcit=max(tpc)
recode targetpostpat targetpostcit (.=0)

drop id tname patents patentcites year_acquired _merge p tpp tpc
duplicates drop

egen id = group(AcquirorUltimateParent ipc)
ren patentyear year
xtset id year
order id AcquirorUltimateParent ipc year

label var targetallpat "all target patents"
label var targetpostpat "count of target patents if post acquisition year"

* patents of acq and ALL targets in a tech/year (add up even before acq)
gen totpat = patentsacq + targetallpat
gen totcit = patentcitesacq + targetallcit
encode Acqui, gen(acq)
drop Acqui

* add #inventors from ipc panel
preserve
	use data/ipc8_patents, clear
	keep ipc inventorcount* largestshare
	duplicates drop
	tempfile ipcinv
	save `ipcinv'
restore

merge m:1 ipc_class_symbol using `ipcinv', keep(3) nogen

compress
xtset id year
save data/acquirer_patents, replace	
	
	
	
	****************************************************************************
	* IPC8 level panel
	****************************************************************************

*381m obs, 33,704,122 patents
odbc load, exec("SELECT tls207_pers_appln.appln_id, tls201_appln.earliest_filing_year, tls201_appln.nb_citing_docdb_fam, tls209_appln_ipc.ipc_class_symbol, tls206_person.han_id FROM tls207_pers_appln INNER JOIN tls201_appln ON tls201_appln.appln_id=tls207_pers_appln.appln_id INNER JOIN tls209_appln_ipc ON tls209_appln_ipc.appln_id=tls207_pers_appln.appln_id INNER JOIN tls206_person ON tls206_person.person_id=tls207_pers_appln.person_id WHERE tls201_appln.appln_kind = 'A' AND tls201_appln.granted = 'Y' ") clear

cap save d:\allpats, replace

 odbc load, exec(" SELECT tls201_appln.appln_id, tls211_pat_publn.publn_date AS CITINGdate, tls211_pat_publn_1.publn_date AS CITEDdate FROM tls201_appln INNER JOIN tls211_pat_publn ON tls201_appln.appln_id = tls211_pat_publn.appln_id INNER JOIN tls212_citation ON tls211_pat_publn.pat_publn_id = tls212_citation.pat_publn_id INNER JOIN tls211_pat_publn AS tls211_pat_publn_1 ON tls212_citation.cited_pat_publn_id = tls211_pat_publn_1.pat_publn_id  INNER JOIN tls201_appln AS cited_application ON tls211_pat_publn_1.appln_id = cited_application.appln_id WHERE  tls211_pat_publn.publn_date < '9999-12-31' AND tls211_pat_publn_1.publn_date < '9999-12-31' AND tls201_appln.granted = 'Y' AND tls201_appln.appln_kind = 'A' AND tls201_appln.appln_id=55749980  ") clear
cap save allcits, replace

* dl all own (self) citations, ie same person id on citer and citee
odbc load, exec(" SELECT tls201_appln.appln_id, tls211_pat_publn.publn_date AS citingdate, tls211_pat_publn_1.publn_date AS citeddate FROM tls201_appln INNER JOIN tls211_pat_publn ON tls201_appln.appln_id = tls211_pat_publn.appln_id INNER JOIN tls212_citation ON tls211_pat_publn.pat_publn_id = tls212_citation.pat_publn_id INNER JOIN tls211_pat_publn AS tls211_pat_publn_1 ON tls212_citation.cited_pat_publn_id = tls211_pat_publn_1.pat_publn_id  INNER JOIN tls201_appln AS cited_application ON tls211_pat_publn_1.appln_id = cited_application.appln_id INNER JOIN tls207_pers_appln ON tls201_appln.appln_id = tls207_pers_appln.appln_id INNER JOIN tls207_pers_appln AS cited ON cited_application.appln_id = cited.appln_id WHERE  tls211_pat_publn.publn_date < '9999-12-31' AND tls211_pat_publn_1.publn_date < '9999-12-31' AND tls201_appln.granted = 'Y' AND tls201_appln.appln_kind = 'A' AND cited.person_id = tls207_pers_appln.person_id  ") clear
* we merge m:1 later
duplicates drop
save data/owncits, replace

use D:\tech_mergers/data/allcits, clear

* remove own citations
merge m:1 appln_id citingdate citeddate using data/owncits
keep if _m==1

gen diff=citingdate-citeddate
drop if diff>365*5
drop ci* diff _merge
bysort appln_id: gen cite_5y=_N
duplicates drop
save data/cite_5y, replace

************************
* compile panel

cd D:\tech_mergers
use data/allpats, clear

* add 5y citations corrected for self citations
merge m:1 appln_id using data/cite_5y, keep(1 3) nogen
recode cite_5y .=0

drop if length(ipc_class_symbol)==4 
replace ipc_class_symbol = subinstr(ipc_class_symbol, " ", "", .)
replace ipc_class_symbol = subinstr(ipc_class_symbol, "/", "", .)
ren earliest_filing_year year
bysort ipc year: gen totpat=_N
bysort ipc year: egen totcit=total(nb_citing_docdb_fam)
bysort ipc year: egen totcit5y=total(cite_5y)

qui unique han_id, by(ipc) generate(inventorcount)
sort ipc inventorcount
by ipc: replace inventorcount=inventorcount[_n-1] if inventorcount==.

* calculate 'market shares' of inventors in ipc classes
bysort ipc_class_symbol han_id: gen count_inv=_N
bysort ipc_class_symbol: gen count_ipc=_N
bysort ipc_class_symbol: egen largestshare=max(count_inv/count_ipc)
drop appln_id nb_citing_docdb_fam han_id count_inv count_ipc cite_5y

duplicates drop
drop if year==9999 | year<1980
egen id=group(ipc)
xtset id year
tsfill, full
bysort id (ipc): replace ipc=ipc[_N]
recode tot* inventor* largest* (.=0)

foreach var in inventorcount largestshare {
	tempvar a
	bysort ipc_class_symbol: egen `a' = max(`var')
	replace `var'=`a'
	drop `a'
	}

compress

preserve
	use data\target_patents, clear
	
	* REVISION IJIO: LIMIT TREATMENT TO OVERLAP
	* adjust treatment to just overlap between venture capital and patent data
	foreach x in .COM ( ) , .AI . - {
		replace tname=subinstr(tname, "`x'", "", .)
		}

	gen target=lower(subinstr(tname, " ", "", .))
	merge m:1 target using data/matched_mergers, keep(3)
	
	keep ipc_class_symbol year Acq tname
	duplicates drop
	*ren year_acquired year
	tempfile a 
	save `a'
restore 

joinby ipc_class_symbol year using `a', unmatched(both)

bysort ipc_class_symbol year: gen N=_N

replace tname="multiple targets" if N>1

bysort ipc year: egen acq_AP= max(AcquirorUltimateParent=="APPLE")
bysort ipc year: egen acq_MI= max(AcquirorUltimateParent=="MICROSOFT")
bysort ipc year: egen acq_FA= max(AcquirorUltimateParent=="FACEBOOK")
bysort ipc year: egen acq_GO= max(AcquirorUltimateParent=="ALPHABET")
bysort ipc year: egen acq_AM= max(AcquirorUltimateParent=="AMAZON")
gen totacq=acq_AP +acq_MI +acq_FA +acq_GO +acq_AM

drop _merge AcquirorUltimateParent N
duplicates drop

sort ipc year
by ipc: replace acq_AP=sum(acq_AP)
by ipc: replace acq_MI=sum(acq_MI)
by ipc: replace acq_FA=sum(acq_FA)
by ipc: replace acq_GO=sum(acq_GO)
by ipc: replace acq_AM=sum(acq_AM)

bysort ipc: gen post=sum(totacq)
gen post_bin=post>0
bysort ipc: egen treated=max(post_bin)

* keep all ipcs
// preserve
// 	use data\acquirer_patents, clear
// 	keep ipc
// 	duplicates drop
// 	tempfile gafamipc
// 	save `gafamipc'
// restore
//
// * keep ipc8 classes without gafam patents as potential controls
// merge m:1 ipc_class_symbol using `gafamipc' // GAFAM are active in 10144 out of 74934 ipc classes
// keep if _m==3 | _m==1
// drop _merge

compress
xtset id year

* ipc8_patents contains all targets with patents; ipc8_patents_overlap only those for which crunchbase data are available

*save data\ipc8_patents, replace		
save data\ipc8_patents_overlap, replace		

	****************************************************************************

* gen graph of GAFAM patents over time
use "D:\tech_mergers\data\acquirer_patents.dta", clear
drop id ipc_class_symbol patentcitesacq post treated targetallpat targetallcit targetpostpat targetpostcit totpat totcit  inventorcount largestshare
bysort acq year: egen totpat=total(patentsacq)
drop patentsacq
duplicates drop
tw (line totpat year if acq==1, lp(solid) ytit("") xtit("")) (line totpat year if acq==2, lp(dash)) (line totpat year if acq==3, lp(dot)) (line totpat year if acq==4, lp(shortdash)) (line totpat year if acq==5, lp(longdash)), legend(ring(0) pos(11) col(1) label(1 "Google") label(2 "Amazon") label(3 "Apple") label(4 "Facebook") label(5 "Microsoft") order(1 3 4 2 5)) graphregion(color(white))
graph export tex/fig/pat_gafam.png, as(png) replace	

* GAFAM patent similarity over time
use "D:\tech_mergers\data\acquirer_patents.dta", clear

bysort acq year: egen totpatgafam=total(patentsacq)

gen share_class=patentsacq/totpatgafam
gen share_alpha=share_class if acq==1
gen share_ama=share_class if acq==2
gen share_apple=share_class if acq==3
gen share_face=share_class if acq==4
gen share_micro=share_class if acq==5

foreach var in share_alpha share_ama share_apple share_face share_micro {
	bysort ipc year: ereplace `var' = max(`var')
	}
recode share_* (.=0)

* calc yearly avg corr (focal vs other 4) plus yearly max corr (focal vs highest corr)
	
drop id patentsacq patentcitesacq post treated targetallpat targetallcit targetpostpat targetpostcit totpat totcit acq inventorcount largestshare totpatgafam share_class	
	
duplicates drop	

egen id=group(ipc_class_symbol)
xtset id year

gen meancorr_alpha=.
gen maxcorr_alpha=.
forvalues i=1990(1)2021 {
	corr share_alpha share_ama share_apple share_face share_micro if year==`i'
	mat A=r(C)
	forvalues b=2(1)5 {
		if A[1,`b']==. mat A[1,`b']=0	
		}
	replace meancorr_alpha=(A[1,2]+ A[1,3]+ A[1,4]+ A[1,5])/4 if year==`i'
	replace maxcorr_alpha=max(A[1,2], A[1,3], A[1,4], A[1,5]) if year==`i'
	}

gen meancorr_ama=.
gen maxcorr_ama=.
forvalues i=1990(1)2021 {
	corr share_ama share_alpha share_apple share_face share_micro if year==`i'
	mat A=r(C)
		forvalues b=2(1)5 {
		if A[1,`b']==. mat A[1,`b']=0	
		}
	replace meancorr_ama=(A[1,2]+ A[1,3]+ A[1,4]+ A[1,5])/4 if year==`i'
	replace maxcorr_ama=max(A[1,2], A[1,3], A[1,4], A[1,5]) if year==`i'
		}

gen meancorr_apple=.
gen maxcorr_apple=.
forvalues i=1990(1)2021 {
	corr share_apple share_ama share_alpha share_face share_micro if year==`i'
	mat A=r(C)
		forvalues b=2(1)5 {
		if A[1,`b']==. mat A[1,`b']=0	
		}
	replace meancorr_apple=(A[1,2]+ A[1,3]+ A[1,4]+ A[1,5])/4 if year==`i'
	replace maxcorr_apple=max(A[1,2], A[1,3], A[1,4], A[1,5]) if year==`i'
		}

gen meancorr_face=.
gen maxcorr_face=.
forvalues i=1990(1)2021 {
	corr share_face share_apple share_ama share_alpha share_micro if year==`i'
	mat A=r(C)
		forvalues b=2(1)5 {
		if A[1,`b']==. mat A[1,`b']=0	
		}
	replace meancorr_face=(A[1,2]+ A[1,3]+ A[1,4]+ A[1,5])/4 if year==`i'
	replace maxcorr_face=max(A[1,2], A[1,3], A[1,4], A[1,5]) if year==`i'
		}	
		
gen meancorr_micro=.
gen maxcorr_micro=.
forvalues i=1990(1)2021 {
	corr share_micro share_face share_apple share_ama share_alpha if year==`i'
	mat A=r(C)
		forvalues b=2(1)5 {
		if A[1,`b']==. mat A[1,`b']=0	
		}
	replace meancorr_micro=(A[1,2]+ A[1,3]+ A[1,4]+ A[1,5])/4 if year==`i'
	replace maxcorr_micro=max(A[1,2], A[1,3], A[1,4], A[1,5]) if year==`i'
		}

drop ipc_class_symbol share_alpha share_ama share_apple share_face share_micro id
duplicates drop	

replace meancorr_face=0 if year<2004 & meancorr_face!=.
replace meancorr_alpha=0 if year<1998 & meancorr_alpha!=.
replace meancorr_ama=0 if year<1994 & meancorr_ama!=.
replace maxcorr_face=0 if year<2004 & maxcorr_face!=.
replace maxcorr_alpha=0 if year<1998 & maxcorr_alpha!=.
replace maxcorr_ama=0 if year<1994 & maxcorr_ama!=.

* replace 2021 with 2020 values due to low sample size
foreach var in meancorr_alpha maxcorr_alpha meancorr_ama maxcorr_ama meancorr_apple maxcorr_apple meancorr_face maxcorr_face meancorr_micro maxcorr_micro {
	replace `var'=`var'[_n-1] if year==2021
}

save data\patent_similarity, replace		

***
use data\patent_similarity, clear

merge m:1 year using data/product_market_similarity_gafam

recode maxcorr_* max_sim_* (.=0)
gen meanpatsim=(maxcorr_alpha+ maxcorr_ama+ maxcorr_apple+ maxcorr_face+ maxcorr_micro)/5
gen meanprodsim=(max_sim_alpha +max_sim_amazon +max_sim_face +max_sim_apple +max_sim_microsoft)/5
recode maxcorr_* max_sim_* (0=.)


tw (line maxcorr* year if year>1993 & year<2020, lp(solid dash longdash dash_dot shortdash) lw(thin)) (line meanpatsim year if year>1993 & year<2020, lw(thick) lc(black)), legend(col(3) label(1 Alphabet) label(2 Amazon) label(3 Apple) label(4 Facebook) label(5 Microsoft) label(6 Mean) ring(0) pos(5) ) xtit("") graphregion(color(white)) name(patsim, replace) title("Patent similarity")

tw (line max_sim_alpha max_sim_amazon max_sim_apple max_sim_face max_sim_microsoft year if year>1993 & year<2019, lp(solid dash longdash dash_dot shortdash)) (line meanprodsim year if year>1993 & year<2019, lw(thick) lc(black)), legend(col(1) ring(0) pos(11)) xtit("") graphregion(color(white)) name(prodsim, replace) title("Product similarity")

grc1leg patsim prodsim, graphregion(color(white)) 
graph export tex/fig/gafam_sim.png, as(png) replace	

* gen graph of GAFAM M&A over time
use "D:\tech_mergers\data\deals.dta", clear
keep year AcquirorUltimateParent
bysort AcquirorUltimateParent year: gen totdeal=_N
encode AcquirorUltimateParent, gen(acq)
duplicates drop
*drop if year<1990
drop if year==2020
tw 	(line totdeal year if acq==1, lp(solid) ytit("") xtit("")) ///
	(line totdeal year if acq==3, lp(dash)) ///
	(line totdeal year if acq==4, lp(dot)) ///
	(line totdeal year if acq==2, lp(shortdash)) ///
	(line totdeal year if acq==5, lp(longdash)), ///
	legend(ring(0) pos(11) col(1) label(1 "Google") label(2 "Apple") label(4 "Amazon")  label(3 "Facebook") label(5 "Microsoft") ) graphregion(color(white)) xscale(range(1985(5)2020)) xlabel(1985(5)2020)
graph export tex/fig/deal_gafam.png, as(png) replace	

	****************************************************************************
	* Matching
	****************************************************************************
		
clear
cap set maxvar 120000
*use data\ipc8_patents, clear
use data\ipc8_patents_overlap, clear

xtset id year
* 306 targets affected in 5665 IPC classes
gen acquisition = treated & !post & f.post

	* selection into treatment: # patent holder in an ipc; level and trend of patents and cites
* means also include post-acquisition patents, ie pot. treatment effects; use running sums instead
bysort id: gen sumpat=sum(totpat)
bysort id: gen sumcit5y=sum(totcit5y)

bysort id: gen patgrowth=(totpat-l.totpat)/l.totpat if !post
replace patgrowth=. if patgrowth==-1 | patgrowth>10
bysort id: egen trend=mean(patgrowth)

gen lninv=log(inventorcount)
gen lnshare=log(largestshare)
gen lnsumpat=ln(sumpat)
gen lnsumcit=ln(sumcit)
gen str4 ipc4=ipc_class_symbol
egen ipc4groups=group(ipc4)
gen lntotpat=ln(totpat+1)
gen lntotcit=ln(totcit5y+1)
su totcit5y,d
gen highcit=totcit5>r(p90)

bysort ipc4: egen mintreat=min(treated) // no controls in ipc4 if == 1
bysort ipc4: egen maxtreat=max(treated) // no treated in ipc4 if == 0
drop if mintreat==1 | maxtreat==0

gen out = (treated & !acquisition) 
global matchcov lntotpat lntotcit lninv lnshare highcit  patgrowth
eststo logit: logit acquisition $matchcov i(1986 1988 1991 1993/2019).year i.ipc4groups if !out

esttab logit using tex/fig/logit.tex, tex star(* 0.1 ** 0.05 *** 0.01) se b(%8.3f) align(dd) compress wide fragment nomtit replace drop(_cons *year* *ipc*) stats(N r2_p, labels("Observations" "Pseudo $ R^2$ ") fmt(%6.0f %4.3f)) label  prehead(\begin{comment}) posthead(\end{comment}) coeflabel(lntotpat "log(Patents)" lntotcit "log(5-year citations)" lninv log(Inventors) lnshare "Largest-patentor markt share" highcit "High citation IPC" patgrowth "Patent growth") nodepvar nomtit

count if acqui & e(sample)
predict phat

su phat if treated, d
su phat if !treated, d
global max = r(max)
	// obs off the common support
count 	if treated & phat!=. & phat>$max
* 1 treatment off common support
bysort id: egen offsupport=max(treated & phat!=. & phat>$max)
tab acquisition if offsupport
drop if offsupport	
foreach var of varlist acq_AP acq_MI acq_FA acq_GO acq_AM {
	gen `var'_bin=(`var'>0)
	}


* pre-matching differences between treated and control group
*global matchcov treathat totalassetstheuryear CNPatCite_g_m NPatCite_g_m innov tfp tfpsq markup markupsq 
cap mat drop x
qui foreach var in $matchcov {
 	loc var2=subinstr(`"`var'"', char(34), "", .)
	di "`var2'"
	ttest `var2' if !out, by(treated)
	mat x=(nullmat(x)\ (r(mu_2), r(mu_1), r(p)) )
	}
mat rownames x = $matchcov  
mat colnames x = meantreat meancontrol p
mat li x

* calculate initial biases
qui foreach var in $matchcov  {
	loc var2=subinstr(`"`var'"', char(34), "", .)
	noi di "`var2'"
	su `var2' if !treated
	local c = r(Var)
	local d = r(mean)
	su `var2' if treated & !out
	* standardized difference in means
	local b = (r(mean) - `d')/(sqrt((r(Var)+`c')/2) )
	noi di "std diff: `b'"
	gen bias`var2'=`b' if treated
	}
	
	* no IPC class constraint, no size crit (but match on trend quartile), no replacement
	*			ADDED TREND and totpat QUARTILES	
	
* matching loop	
gen control=0
gen matchedto=0
gen pscore_diff=0
*drop t
gen t=0 if acquisition
sort id year
gen str3 ipc3=ipc_class_symbol
// xtile qtrend=trend, nq(4)
xtile qpat=totpat, nq(4)

global N = _N
loc controlcount=0
qui forvalues i=1/ $N {
	if t[`i']==0 & treated[`i']==1 & sumpat[`i']!=0 {
	    
		*noi di `i'
		cap drop sortvar
		cap drop pdiff
		
		*loc ipc=ipc3[`i']
		loc y=year[`i']
		loc id=id[`i']
		loc qp=qpat[`i']
		
		loc pscore=phat[`i']
		gen sortvar=(year==`y' & !treated & phat!=. & qpat==`qp')
		// if no within-ind matches, relax constraint: 5-digit ipc
		count if sortvar 
			if r(N)==0   noi di "ERROR: no match found"
			if r(N)==0   continue
		
		gen double pdiff=abs(phat - `pscore') if sortvar
		su pdiff, meanonly
		loc min = r(min)
		su id if float(pdiff)<=float(`min') & sortvar
		loc match = r(min)
		if r(min)!=r(max)	 noi di "ERROR: double match, use lower id"
		noi di `i' " matched id: " `match'
		
		su control if id==`match'
		
		if r(mean)>0 {
			noi di "multiple control"
			preserve 	
			keep if id==`match'
			replace id=100000+`controlcount'
			loc controlcount=`controlcount'+1
			replace t=.
			tempfile add
			save `add', replace
			
			* update the loc that ids control
			su id 
			loc match = r(min)

			restore
			replace t=. if id==`match'
			append using `add'
			}
		
		replace t=0 if id==`match' & year==`y'
		replace control=control+1 if id==`match'
		replace matchedto=`match' if id==`id'
		replace pscore_diff=float(`min') if id==`id'
			
		drop sortvar pdiff
		sort id year
		}
	}	

count if acqui & pscore_diff!=0
count if acqui


*************	do the 1vs1 regs here, before dropping pairs, to maximize obs

save matchtempoverlap, replace
preserve

		* pairwise effects for comparison with investment effects		
	cap drop coeff tstat N
	gen tstat=.
	gen coeff=.	
	gen coeffzero=.
	gen N=.
	
	keep if (treated & matchedto!=0) | control	
	drop if matchedto==.
	
	gsort id -year
	bysort id: replace t=t[_n-1]-1 if t==.
	sort id year
	bysort id: replace t=t[_n-1]+1 if t==.
	cap drop post
	gen post=t>0
	
	xtset id t

	gen lntotcit5yzero=ln(totcit5y)

qui	levelsof id if treated, local(treat)
qui	 foreach id of local treat {
		levelsof matchedto if id==`id', local(id2)
		cap	reghdfe lntotpat 1.treated##1.post  if id==`id' | id==`id2', absorb(id year) keepsingletons
		noi di _b[1.treated#1.post]
		replace coeff=_b[1.treated#1.post] if id==`id'
		replace tstat=_b[1.treated#1.post]/_se[1.treated#1.post] if id==`id'
		replace N=e(N) if id==`id'
		}

	cap drop coeff2 tstat2 N2
gen tstat2=.
gen coeff2=.			
gen N2=.
	
levelsof tname, local(targets)
foreach name of local targets {
	count if tname=="`name'"
	if r(N)>250 continue 
	
 	levelsof id if tname=="`name'", local(id1) separate(,)
 	levelsof matchedto if tname=="`name'", local(id2) separate(,)
	
	gen sample = inlist(id, `id1')
	replace sample = max(sample, inlist(id, `id2'))
	
	reghdfe lntotpat 1.treated##1.post if sample, absorb(id year) keepsingletons
	noi di _b[1.treated#1.post]
	replace coeff2=_b[1.treated#1.post] if sample
	replace tstat2=_b[1.treated#1.post]/_se[1.treated#1.post] if sample
	replace N2=e(N) if sample
	drop sample
	}	
		
		
		
		
	merge m:1 tname using data/deals, keepusing(AcquirorUltimateParent) keep(1 3) nogen
	keep tname ipc_class coeff* tstat* year AcquirorUltimateParent
	keep if tname!=""
		
	bysort  tname year: egen mcoeff=mean(coeff)
	bysort  tname year: egen mcoeff2=mean(sign(coeff))
	bysort  tname year: egen mcoeff_deal=mean(coeff2)
	bysort  tname year: egen mcoeff2_deal=mean(sign(coeff2))

*	bysort  tname year: egen mtstat=mean(tstat)
	drop ipc_class tstat* coeff*
	duplicates drop
	
	drop if regexm(tname, "multiple targets")
	
	*ren (mcoeff mcoeff2 mcoeff3) (mc_notrend mc_notrend2 mc_notrend3)
	*merge 1:1 tname using data\cit_by_case_new2, nogen
	
	save data\cit_by_case_psm, replace
	

	*************	do the 1vs1 regs here, before dropping pairs, to maximize obs

restore

	* PS densities before matching
twoway (kdensity phat if treated) (kdensity phat if !treated, ytit("") xtit("") title("Before matching") lpattern(dash)) if phat<.1, graphregion(color(white)) legend(off) name(before, replace) 	
	
keep if (treated & matchedto!=0) | control	
drop if matchedto==.
compress
* 5374 treated/control pairs, 10748 ids -> need to drop some for balance
* overlap: 3411 treated
	
	* PS densities after matching	
twoway (kdensity phat if treated & t==0) (kdensity phat if !treated & t==0, ytit("") title("After matching") lpattern(dash)) if phat<.1, legend(col(2) label( 1 "Treated") label( 2 "Control" ) ) graphregion(color(white)) xtit("") name(after, replace) 
	
grc1leg	before after, legendfrom(after) graphregion(color(white)) 
graph export "tex/fig/beforeafter.png", as(png) replace	

cap mat drop y
qui foreach var in $matchcov {
 	loc var2=subinstr(`"`var'"', char(34), "", .)
	di "`var2'"
	ttest `var2' if t==0, by(treated)
	mat y=(nullmat(y)\ (r(mu_2), r(mu_1), r(p)) )
	}
mat rownames y = $matchcov  
mat colnames x = meantreat meancontrol 
mat li y

mat z = x, y
mat rownames z = Patents   "5-year citations"    Inventors    C1   " High citation IPC"      "Patent growth"
mat li z
esttab matrix(z, fmt(2 2 2 2 2 2)) using tex/fig/before_after.tex, noobs nonum nomtit replace b(%4.4f) label prehead("\begin{comment}") posthead("\end{comment}") nogaps  prefoot("") postfoot("")

* calculate biases after matching
qui foreach var in $matchcov  {
 	loc var2=subinstr(`"`var'"', char(34), "", .)
	di "`var2'"
	su `var2' if !treated & t==0 
	local d = r(mean)
	qui su `var2' if !treated
	local c = r(Var)
	su `var2' if treated & t==0
	* standardized difference in means
	local b = (r(mean) - `d')/( sqrt((r(Var)+`c')/2) )
	local b = abs(`b')
	noi di "std diff: `b'"
	replace bias`var2'=`b' if _n==1
	*qui replace bias`var'=0.02 if bias`var'<0.02 & _n==1
	* sig of stand diff
	qui reg `var2' treated if t==0
	qui test treated=0
	noi di "p-value: `r(p)'"
	}

sort id year	
gen matching=0
replace matching=1 if _n==1
label define prepost 0 "Before Matching" 1 "After Matching"
label values matching prepost

twoway connect bias* matching, yline(0, lw(medthick) lc(black)) ytitle(Standardized Bias) xtitle("") graphregion(color(white)) ms(O D T S X + Oh Dh Th Sh o d s) ///
	lp(solid dash dot dash_dot shortdash shortdash_dot longdash longdash_dot) xscale(range(-0.15 1.15)) xlabel(0(1)1 ,val) yscale(r(-0.5(.2)1.4)) ///
	ylabel(-0.5(.25)1.45) graphregion(color(white)) legend(ring(0) label(1 Patents) label(2 "5-year citations") label(3 Inventors) label(4 "C1") label(5 "High citation IPC")  label(6 "Patent growth") pos(1) cols(1) textfirst rowgap(.5) symxsize(5) region(lstyle(none)) size(small) region(lwidth(none))) name(biasred, replace)	
graph export "tex/fig/biasred.png", as(png) replace
drop bias*

gsort id -year
bysort id: replace t=t[_n-1]-1 if t==.
sort id year
bysort id: replace t=t[_n-1]+1 if t==.
cap drop post
gen post=t>0

xtset id t

drop mintreat maxtreat out matching offsupport ipc3 sumpat sumcit patgrowth ipc4
drop inventorcount largestshare lninv lnshare lnsumpat lnsumcit lntotpat lntotcit
drop trend highcit  qpat post_bin
cap drop *pweigh*
est clear
order id year
compress

gen lntotcit5yzero=ln(totcit5y)
gen lntotpat=ln(totpat)

gen treat_2010 = (tname!="" & year<=2010)
gen treat_2020 = (tname!="" & year>2010 & year<=2020)
bysort id: ereplace treat_2010 = max(treat_2010)
bysort id: ereplace treat_2020 = max(treat_2020)
gen c=1-treated
gen early=t<-10
gen late=t>10
gen early2 = t<-5
gen late2 = t>5
gen treatpost=treated*post
gen treatpost2010=treat_2010*post
gen treatpost2020=treat_2020*post
gen treatpostAP=treated*post*acq_AP
gen treatpostMI=treated*post*acq_MI
gen treatpostFA=treated*post*acq_FA
gen treatpostGO=treated*post*acq_GO
gen treatpostAM=treated*post*acq_AM
foreach x in AP MI FA GO AM {
	gen treatpost`x'2010=treatpost`x'*treatpost2010
	gen treatpost`x'2020=treatpost`x'*treatpost2020
}

save data\ipc8_patents_matched_overlap, replace



	****************************************************************************
	* Estimation
	****************************************************************************

global regopt absorb(id year) keepsingletons cl(ipc4)
global output tex align(dd) fragment replace nodep nomtit prehead(\begin{comment}) posthead(\end{comment})
global format1 compress star(* 0.1 ** 0.05 *** 0.01) se b(%9.3f) stats(N r2, labels("Observations" "$ R^2$ ") fmt(%6.0f %4.3f)) nonotes nonum label rename(1.treated#1.post "M\&A" 1.treated#1.post#1.treat_2010 "M\&A (1990-2010)" 1.treated#1.post#1.treat_2015 "M\&A_2015"  1.treated#1.post#1.treat_2020 "M\&A (2011-2020)" 1.post#1.acq_AP_bin Apple 1.post#1.acq_MI_bin Microsoft 1.post#1.acq_FA_bin Facebook 1.post#1.acq_GO_bin Google 1.post#1.acq_AM_bin Amazon) order("M\&A" "M\&A_2010" "M\&A_2020" Google Apple Facebook Amazon Microsoft)
global format2 compress star(* 0.1 ** 0.05 *** 0.01) se b(%9.3f) stats(N r2, labels("Observations" "$ R^2$ ") fmt(%6.0f %4.3f)) nonotes nonum label rename(treatpost "M\&A" treatpostAP Apple treatpostMI Microsoft treatpostFA Facebook treatpostGO Google treatpostAM Amazon)

foreach sample in all overlap {	
	
if "`sample'"=="all" use data\ipc8_patents_matched, clear
else use data\ipc8_patents_matched_overlap, clear

bysort id: egen cohort=max(year*acquisition)
replace cohort=. if !treated

* pretrends, sun and abraham estimator
if "`sample'"=="all" {
	*eststo sunab: eventstudyinteract lntotpat early t_10-t_2 t0 t1 - t20 , absorb(id year) cohort(cohort) control_cohort(c) vce(cluster ipc4)
	forvalues i = -39(1)34 {
	loc j = abs(`i')
	if `i'>=0 gen t`j' = (treated*t==`i') 
	if `i'<0 gen t_`j'= (treated*t==`i') 
	}
	eststo sunab: eventstudyinteract lntotpat early2 t_5-t_2 t0 t1 - t20 , absorb(id year) cohort(cohort) control_cohort(c) vce(cluster ipc4)
	matrix C = e(b_iw)
	mata st_matrix("A",sqrt(diagonal(st_matrix("e(V_iw)"))))
	matrix C = C \ A'
	matrix list C
	coefplot matrix(C[1]), se(C[2]) vertical keep(t_5 t_4 t_3 t_2 t0 t1 t2 t3 t4 t5) rename(t_5 = "t=-5" t_4 = "t=-4" t_3 = "t=-3" t_2 = "t=-2" t0="t=0" t1 = "t=1" t2 = "t=2" t3 = "t=3" t4 = "t=4" t5 = "t=5") xline(4.5) yline(0)  graphregion(color(white)) 
	graph rename Graph pretrends, replace
	graph save pretrends "tex/fig/pretrends_pat.gph", replace
	graph export "tex/fig/pretrends_pat.png", as(png) replace
	}

* main results
foreach var in lntotcit5yzero {		
	eventstudyinteract `var' treatpost, absorb(id year) cohort(cohort) control_cohort(c) vce(cluster ipc4)
	matrix b_iw = e(b_iw)
	matrix V_iw = e(V_iw)
	erepost b = b_iw V = V_iw
	eststo `var'`sample'1
	
	eventstudyinteract `var' treatpost????, absorb(id year) cohort(cohort) control_cohort(c) vce(cluster ipc4)
	matrix b_iw = e(b_iw)
	matrix V_iw = e(V_iw)
	erepost b = b_iw V = V_iw
	eststo `var'`sample'2
	
	eventstudyinteract `var' treatpost??, absorb(id year) cohort(cohort) control_cohort(c) vce(cluster ipc4)
	matrix b_iw = e(b_iw)
	matrix V_iw = e(V_iw)
	erepost b = b_iw V = V_iw
	eststo `var'`sample'3
	}

*esttab lntotcit5yzero?, 		$format2
esttab lntotpat`sample'?, 		$format2 
esttab lntotcit5yzero?	using tex/fig/lntotcit5yzero_matched.tex, $format2 $output wide
esttab lntotpat`sample'?	using tex/fig/lntotpat_matched_`sample'.tex, $format2 $output 

* acq/period het
eventstudyinteract lntotpat treatpost??????, absorb(id year) cohort(cohort) control_cohort(c) vce(cluster ipc4)
matrix b_iw = e(b_iw)
matrix V_iw = e(V_iw)
erepost b = b_iw V = V_iw
eststo het_`sample'

esttab het_`sample', order(*Google* *Apple* *Face* *Ama* *Mic*) rename(treatpostGO2010 "Google 2010" treatpostGO2020 "Google 2020" treatpostAP2010 "Apple 2010" treatpostAP2020 "Apple 2020" treatpostFA2010 "Facebook 2010" treatpostFA2020 "Facebook 2020" treatpostAM2010 "Amazon 2010" treatpostAM2020 "Amazon 2020" treatpostMI2010 "Microsoft 2010" treatpostMI2020 "Microsoft 2020") wide se

esttab het_`sample' using tex/fig/het_`sample'.tex,  $output order(*Google* *Apple* *Face* *Ama* *Mic*) rename(treatpostGO2010 "Google 2010" treatpostGO2020 "Google 2020" treatpostAP2010 "Apple 2010" treatpostAP2020 "Apple 2020" treatpostFA2010 "Facebook 2010" treatpostFA2020 "Facebook 2020" treatpostAM2010 "Amazon 2010" treatpostAM2020 "Amazon 2020" treatpostMI2010 "Microsoft 2010" treatpostMI2020 "Microsoft 2020") wide compress star(* 0.1 ** 0.05 *** 0.01) se b(%9.3f) stats(N r2, labels("Observations" "$ R^2$ ") fmt(%6.0f %4.3f)) nonotes nonum label  se



* similarity of GAFAM firms over time

* patent portfolio
merge m:1 year using data/patent_similarity, nogen
* stack the individual mean/max correlations into one var
gen patsim_mean = max(meancorr_alpha*acq_GO, meancorr_ama*acq_AM, meancorr_apple*acq_AP, meancorr_face*acq_FA, meancorr_micro*acq_MI)
gen patsim_max = max(maxcorr_alpha*acq_GO, maxcorr_ama*acq_AM, maxcorr_apple*acq_AP, maxcorr_face*acq_FA, maxcorr_micro*acq_MI)

* business descriptions
merge m:1 year using data/product_market_similarity_gafam, nogen
gen prodsim_mean = max(mean_sim_alpha*acq_GO, mean_sim_amazon*acq_AM, mean_sim_apple*acq_AP, mean_sim_face*acq_FA, mean_sim_microsoft*acq_MI)
gen prodsim_max = max(max_sim_alpha*acq_GO, max_sim_amazon*acq_AM, max_sim_apple*acq_AP, max_sim_face*acq_FA, max_sim_microsoft*acq_MI)
drop meancorr_* maxcorr_* mean_sim_* max_sim_* m_google
recode patsim* prodsim* (.=0)

eststo sim1_`sample': reghdfe lntotpat 1.treated#1.post 1.treated#1.post#c.patsim_mean 1.treated#1.post#c.prodsim_mean, $regopt
eststo sim2_`sample': reghdfe lntotpat 1.treated#1.post 1.treated#1.post#c.patsim_max 1.treated#1.post#c.prodsim_max, $regopt
eststo sim3_`sample': reghdfe lntotpat 1.treated#1.post#1.treat_* 1.treated#c.patsim_mean#1.post#1.treat_* 1.treated#c.prodsim_mean#1.post#1.treat_*, $regopt
eststo sim4_`sample': reghdfe lntotpat 1.treated#1.post#1.treat_* 1.treated#c.patsim_max#1.post#1.treat_* 1.treated#c.prodsim_max#1.post#1.treat_*, $regopt
* sim times identity
// eststo sim5_`sample': reghdfe lntotpat 1.treated#1.post 1.treated#1.post#1.acq_*_bin#c.*sim_mean , $regopt
// eststo sim6_`sample': reghdfe lntotpat 1.treated#1.post 1.treated#1.post#1.acq_*_bin#c.*sim_max , $regopt

esttab sim2_`sample' sim4_`sample',  wide rename(1.treated#1.post "M\&A" 1.treated#1.post#c.patsim_max "M\&A$ \times$ patsim" 1.treated#1.post#c.prodsim_max "M\&A$ \times$ prodsim" 1.treated#1.post#1.treat_2010 "M\&A_2010" 1.treated#1.post#1.treat_2020 "M\&A_2020" 1.treated#1.post#1.treat_2010#c.patsim_max "M\&A_2010patsim" 1.treated#1.post#1.treat_2020#c.patsim_max "M\&A_2020patsim" 1.treated#1.post#1.treat_2010#c.prodsim_max "M\&A_2010prodsim" 1.treated#1.post#1.treat_2020#c.prodsim_max "M\&A_2020prodsim")

esttab sim2_`sample' sim4_`sample' using tex/fig/sim_`sample'.tex, drop(_cons)  compress star(* 0.1 ** 0.05 *** 0.01) se b(%9.3f) stats(N r2, labels("Observations" "$ R^2$ ") fmt(%6.0f %4.3f)) nonotes nonum label rename(1.treated#1.post "M\&A" 1.treated#1.post#c.patsim_max "M\&A$ \times$ patsim" 1.treated#1.post#c.prodsim_max "M\&A$ \times$ prodsim" 1.treated#1.post#1.treat_2010 "M\&A_2010" 1.treated#1.post#1.treat_2020 "M\&A_2020" 1.treated#1.post#1.treat_2010#c.patsim_max "M\&A_2010patsim" 1.treated#1.post#1.treat_2020#c.patsim_max "M\&A_2020patsim" 1.treated#1.post#1.treat_2010#c.prodsim_max "M\&A_2010prodsim" 1.treated#1.post#1.treat_2020#c.prodsim_max "M\&A_2020prodsim") tex align(dd) fragment replace nodep nomtit prehead(\begin{comment}) posthead(\end{comment})

* 

merge m:1 tname using data/het_did_matchfile

recode bin_common_ipc age_at_acq private (.=0)

gen after2010=(year_acquired>2010)
gen morethanfiveyears=(age>5)
gen morethanoneround=num_funding_rounds>1

reghdfe lntotpat 1.treated#1.post 1.treated#1.post#1.(bin_common private after2010 morethanfiveyears), $regopt

esttab . using tex/fig/pol_`sample'.tex, drop(_cons) compress star(* 0.1 ** 0.05 *** 0.01) se b(%9.3f) stats(N r2, labels("Observations" "$ R^2$ ") fmt(%6.0f %4.3f)) nonotes nonum label rename(1.treated#1.post "M\&A" 1.treated#1.post#c.bin_common_ipc "Common IPC" 1.treated#1.post#1.private "Private deal" 1.treated#1.post#1.after2010 "After 2010" 1.treated#1.post#1.morethanfiveyears "Target age $>$ 5" ) tex align(dd) fragment replace nodep nomtit prehead(\begin{comment}) posthead(\end{comment})

}



bysort t treated: egen mpatt=mean(lntotpat)
sort t
tw (line mpatt t if treated , lc(black) xline(0)) (line mpatt t if !treated , lp(dash) lc(black)) if abs(t)<9, legend(ring(0) pos(11) label(1 "IPC classes with acquisitions") label(2 "IPC classes without acquisitions") rows(2)) ytit("") xtit("") graphregion(color(white))  name(citt, replace)

bysort year treated: egen mpaty=mean(lntotpat)
sort year
tw (line mpaty year if treated , lc(black) xline(0)) (line mpaty year if !treated , lp(dash) lc(black)) if abs(t)<10 & year<2018, legend(ring(0) pos(11) label(1 "IPC classes with acquisitions") label(2 "IPC classes without acquisitions") rows(2)) ytit("") xtit("") graphregion(color(white))  name(yeartrend_pat, replace)
graph save yeartrend_pat "tex/fig/yeartrend_pat.gph", replace

grc1leg  city citt, col(2) graphregion(color(white))
graph export "tex/fig/patcitt.png", as(png) replace



	****************************************************************************
	* pairwise effects for comparison with investment effects
	****************************************************************************
				
use data\ipc8_patents_matched, clear

gen lntotcit5y=ln(totcit5y+1)
gen lntotcit5yzero=ln(totcit5y)
merge m:1 tname using data/deals, keepusing(AcquirorUltimateParent) keep(1 3) nogen
		
cap drop coeff tstat N
gen tstat=.
gen coeff=.			
gen N=.

levelsof id if treated, local(treat)
 foreach id of local treat {
	levelsof matchedto if id==`id', local(id2)
	 reghdfe lntotcit5yzero 1.treated##1.post if id==`id' | id==`id2', absorb(id year) keepsingletons
	noi di _b[1.treated#1.post]
	replace coeff=_b[1.treated#1.post] if id==`id'
	replace tstat=_b[1.treated#1.post]/_se[1.treated#1.post] if id==`id'
	replace N=e(N) if id==`id'
	}

cap drop coeff2 tstat2 N2
gen tstat2=.
gen coeff2=.			
gen N2=.
	
levelsof tname, local(targets)
foreach name of local targets {
	count if tname=="`name'"
	if r(N)>250 continue 
	
 	levelsof id if tname=="`name'", local(id1) separate(,)
 	levelsof matchedto if tname=="`name'", local(id2) separate(,)
	
	gen sample = inlist(id, `id1')
	replace sample = max(sample, inlist(id, `id2'))
	
	reghdfe lntotcit5yzero 1.treated##1.post if sample, absorb(id year) keepsingletons
	noi di _b[1.treated#1.post]
	replace coeff2=_b[1.treated#1.post] if sample
	replace tstat2=_b[1.treated#1.post]/_se[1.treated#1.post] if sample
	replace N2=e(N) if sample
	drop sample
	}
	
	
keep tname ipc_class coeff* tstat* year AcquirorUltimateParent
keep if tname!=""

preserve
	keep if tname=="multiple targets"
	drop tname
	ren year year_acquired
	merge m:m AcquirorUltimateParent ipc_class_symbol year_acquired using data/target_patents, keepusing(tname) keep(3) nogen
	ren year_acquired year
	tempfile add
	save `add', replace
restore

drop if tname=="multiple targets"
append using `add'
	
bysort  tname year: egen coeff_avg=mean(coeff)
bysort  tname year: egen coeff_deal=mean(coeff2)


*bysort  tname year: egen mtstat=mean(tstat)
drop ipc_class tstat tstat2 coeff2 coeff
duplicates drop
	
save data\cit_by_case, replace

	****************************************************************************
	* tech overlap, ie do acq and tar have patents in same ipc8?
	****************************************************************************

use data/acquirer_patents, clear	
decode acq, gen(AcquirorUltimateParent)	
ren year patentyear

* only pre-merger tech overlap
drop if post

* 1:m due to multiple targets
merge 1:m ipc_class_symbol AcquirorUltimateParent patentyear using data/target_patents, keepusing(tname year_acquired) 
drop if tn==""

bysort AcquirorUltimateParent tname: gen targetipcs=_N
bysort AcquirorUltimateParent tname: egen overlapipcs=total(_merge==3)
gen share_commonipc=overlapipcs/targetipcs

keep AcquirorUltimateParent tname year_acquired share_commonipc
duplicates drop

save data/tech_overlap, replace
	
	
	
	
	* distribution of gafam patents across ipc2 classes
	
use "D:\tech_mergers\data\acquirer_patents.dta", clear

gen str2 ipc2=ipc_
egen ipc2num=group(ipc2)

preserve
keep if year<2010 & year>=2000
bysort acq ipc2num: egen countipc2_2010=total(totpat)

tw (bar countipc2_2010 ipc2num if acq==1, barwidth(1.4)) (bar countipc2_2010 ipc2num if acq==2), name(AMA_GO_2010, replace) legend(off)
tw (bar countipc2_2010 ipc2num if acq==1, barwidth(1.4)) (bar countipc2_2010 ipc2num if acq==3), name(APP_GO_2010, replace) legend(off)
tw (bar countipc2_2010 ipc2num if acq==1, barwidth(1.4)) (bar countipc2_2010 ipc2num if acq==4), name(FACE_GO_2010, replace) legend(off)
tw (bar countipc2_2010 ipc2num if acq==1, barwidth(1.4)) (bar countipc2_2010 ipc2num if acq==5), name(MS_GO_2010, replace) legend(off)

restore

keep if year<2020 & year>=2010
bysort acq ipc2num: egen countipc2_2020=total(totpat)

tw (bar countipc2_2020 ipc2num if acq==1, barwidth(1.4)) (bar countipc2_2020 ipc2num if acq==2), name(AMA_GO_2020, replace) legend(off)
tw (bar countipc2_2020 ipc2num if acq==1, barwidth(1.4)) (bar countipc2_2020 ipc2num if acq==3), name(APP_GO_2020, replace) legend(off)
tw (bar countipc2_2020 ipc2num if acq==1, barwidth(1.4)) (bar countipc2_2020 ipc2num if acq==4), name(FACE_GO_2020, replace) legend(off)
tw (bar countipc2_2020 ipc2num if acq==1, barwidth(1.4)) (bar countipc2_2020 ipc2num if acq==5), name(MS_GO_2020, replace) legend(off)

graph combine AMA_GO_2010 AMA_GO_2020, ycommon name(AMAGO, replace)
graph combine APP_GO_2010 APP_GO_2020, ycommon name(APPGO, replace)
graph combine FACE_GO_2010 FACE_GO_2020, ycommon name(FACEGO, replace)
graph combine MS_GO_2010 MS_GO_2020, ycommon name(MSGO, replace)

