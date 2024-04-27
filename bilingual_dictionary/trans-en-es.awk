# gawk script to extract translations from the database dump of en.wiktionary.org
# Version: 20191222
#
# (c) 2011-2019 by Matthias Buchmeier
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
# TODO: inclusion of English "alternative forms/spellings" as trans-see links
# TODO: include blacklist of pages to be excluded, e.g Taumatawhakatangihangakoauauotamateaturipukakapikimaungahoronukupokaiwhenuakitanatahu
# TODO: proper treatment of diacritics following Module:languages. 
#       Currently only diacritics on Russian Arabic, Hebrew are removed. Option to keep diacritics should be implemented.
# TODO: {{unsupported|
#
BEGIN {
#
# target language configuration
# to configure the target language edit the lines in the section "User defined variables"
# or configure the target language on the command-line with the following options:
#
# Command-line options:
#######################
#  required gawk command-line switches:
#
#    name of the language to be extracted, or bar-separated list of languages:
#    -v LANG=language or  -v LANG="language1|language2|language3"
#	
#    iso-639-code of the language to be extracted, or bar-separated list of iso-codes:
#    -v ISO=iso-code   or   -v ISO="iso1|iso2|iso3"
# 
#    name of the language family as specified on the headline of a nested section:
#    (required only if LANG contains multiple languages)
#    -v GENERIC_LANG=language_family
#
#  optional gawk command-line switches:
#
#    this option has to be used for languages written in non-Latin script, e.g. Cyrillic, Greek, etc.:
#    -v LATIN="n"    
#
#    remove wiki-links and wiki-style bolding, italicizing:
#    -v REMOVE_WIKILINKS="y"
#
#    include trans-see links:
#    -v TRANS_SEE="y"  
#
#    include English pronunciation (IPA): 
#    -v ENABLE_IPA="y"
#
#    bar-separated list of languages to be excluded
#    (the default is to include all nested lines):
#    -v EXCLUDE_LANG="language1|language2|language3"
#
#    bar-separated list of qualifiers to be added via iso-code
#    (specified in the same order as the ISO-list):
#    -v ISO_QUALIFIER="qualifier1|qualifier2|qualifier3"
#    
#    bar-separated list of qualifiers to be added via subsection language
#    (specified in the same order as the LANG-list):
#    -v LANG_QUALIFIER="qualifier1|qualifier2|qualifier3"
#
#    don't include transliterations:
#    -v REMOVE_TRANSLIT="y"
#
##########################
# User defined variables:
##########################
#
# English names of the target language as specified at the beginning of translation lines,
# multiple names have to be separated by "|":
# If the translations are nested then this list should include:
#	the language family name and the nested section language names
# for an example of a language with nested translations see the Norwegian section below
# and the Norwegen translation sections
# lang = "Spanish";
#
# iso 639 language codes of the target language as used in the t-templates on the translation lines,
# multiple codes have to separated by "|":
# iso = "es";
#
# unique language family name, as used on the nesting headline
# only required if translations are nested
# generic_lang = "Spanish";
generic_lang = "";
#
# language headwords of nested sections to be excluded from the dictionary
# multiple languages have to separated by "|":
exclude_lang = "";
#
# set to 1/0  for Latin/non-Latin script
latin = 1;
#
# set to 1 if you want to remove wiki-syntax (wiki-links and wiki-style bolding and italicizing)
remove_wikilinks = 0;
#
# set to 1 if transliterations contain wikilinks
links_inside_tr = 0;
#
# show the links of trans-see sections
enable_trans_see = 0;
#
# include only trans-see links with existing target
# this uses trans-see links from a precompiled file (use -v LANG="Trans_See" to build it)
# precompiled trans-see filename (an empty file-name disables this option):
trans_see_file = "trans-see.wiki"; 
#
# include content of translations to be checked (ttbc) sections
# warning: this content is likely less reliable
enable_ttbc = 1;
#
# include English international phonetic alphabet (IPA) pronunciation information
enable_ipa = 0;
#
# remove transliterations
rmtr = 0;
#
# output formatting options:
#############################
#
# separator between the English part and the translations
Sep = "::";
#
# replaces commas separating translations with this string
# it should be a semicolon or empty string (as some translations are separated by semicolon)
# feature is disabled if string is empty
TransSep = "";
# separator between multiple concatenated translation lines (nested translation sections or multiple languages) 
TransLineSep = "; ";
#
# parsing code for command-line switches
#########################################
#
if(LANG!="") lang = LANG;
#
FIXME = "FIXME-en-" lang ".txt";
# experimental feature to get transliteration from lua module
# strangely the lua scripts seems to require lua version 5.1; neither 5.0 nor 5.2 works
if(LUA == "y")  luapipe=1;
	else { luapipe=0; luascript = "";}
#
# preconfigured language options:
##################################
#
#default excluded languages (can be overridden by command-line option)
if(EXCLUDE_LANG == "")
{
if(lang == "French") exclude_lang = "French Creole|Old French|Middle French|Gallo|Norman";
if(lang == "Spanish") exclude_lang = "Old Spanish|Aragonese";
if(lang == "German") exclude_lang = "German Low German|Middle Low German|Low German|Middle High German|Old High German|Alemannic|Alemannic German|Kölsch|Bavarian|Alsatian|Badisch|Berliner|Bernese|Camelottisch|Frankonian|Lichtensteinisch|Luxembourgeois|Moselfraenkisch|Plattdeutsch|Rhoihessisch|Ruhrisch|Saarlaendisch|Saxon|Swabian|Viennese|Alsace|Palatinate German|Swiss German|Kölsch|Silesian German|Saterland|Pennsylvania German|Alemannic|Central Franconian|Rhine Franconian|Alemannic|Cimbrian|Luxembourgish|Sathmar Swabian|East Central German|East Franconian|Silesian|Gottscheerish|Kölnisch|Hochpreußisch|Oberlausitzisch";
if(lang == "Italian") exclude_lang = "Sicilian|Old Italian";
if(lang == "Korean") {exclude_lang = "Old Korean"; luascript = "cd Lua_Modules/ && ./ko-TR.lua";}
if(lang == "Portuguese") exclude_lang = "Old Portuguese";
if(lang == "Polish") exclude_lang = "Old Polish";
if(lang == "Romanian") exclude_lang = "Cyrillic";
if(lang == "Swedish") exclude_lang = "Old Swedish";
if(lang == "Hindi") exclude_lang = "Fiji Hindi";
if(lang == "Vietnamese") exclude_lang = "Chu Nom";
if(lang == "Turkish") exclude_lang = "Ottoman Turkish|Old Turkic|Arabic|Ottoman";
if(lang == "Catalan") exclude_lang = "Old Catalan";
if(lang == "Lithuanian") exclude_lang = "Samogitian";
}
#
# predefined language specific options (uncomment if you want to configure on the command line)
# fake "Trans_See" language used to compile the "trans-see.wiki" file
if(lang == "Trans_See") {
generic_lang = "XXXX";
iso = "XXXX";
enable_trans_see = 1;
trans_see_file = "";
}
#
# Multilingual 
# output translations to all languages tab-separated on one line
# this requires some postprocessing, but will significantly speedup the compilation
if(lang == "Multilingual")
{
LANG = "";
iso = "";
lang = "";
generic_lang = "";
latin = 0;
TransLineSep = "|";
remove_wikilinks = 1;
}
#
if(lang == "Norwegian")
{
generic_lang = "Norwegian";
iso = "no|nn|nb";
lang = "Norwegian|Nynorsk|Norwegian Nynorsk|Bokmål|Norwegian Bokmål|Norwegian Høgnorsk";
iso_qualifier = "|Nynorsk|Bokmål";
lang_qualifier = "|Nynorsk|Nynorsk|Bokmål|Bokmål|Høgnorsk";
exclude_lang = "Old Norse|Old Norwegian"
}
#
if(lang == "Dutch")
{
generic_lang = "Dutch";
iso = "nl|vls";
lang = "Dutch|Flemish|West Flemish|Brabantish";
iso_qualifier = "|Flemish";
lang_qualifier = "|Flemish|Flemish|Brabantish";
exclude_lang = "Dutch Low Saxon|Dutch Low German|Old Dutch|Middle Dutch|Drents|Gronings|Twents|Low German";
}
#
if(lang == "Japanese")
{
iso = "ja";
links_inside_tr = 1;
latin = 0;
}
#
if(lang == "Standard_Arabic")
{
generic_lang = "Arabic";
lang = "Arabic|MSA|Standard Arabic";
iso = "ar|arb";
latin = 0;
exclude_lang = "Algerian|Andalusian|Bahrani|Chadian|Egyptian|Egyptian Arabic|Gulf|Gulf Arabic|Hassānīya|Iraqi|Iraqi Arabic|Lebanese|Lebanese/Syrian|Levantine|Levantine Arabic|Libyan|Moroccan|Moroccan Arabic|Morocco|North Levantine Arabic|Palestinian|Palestinian Arabic|South Levantine Arabic|Syrian|Sudanese|Tunisian Arabic|UAE|Hadrami Arabic|Hijazi Arabic|Juba Arabic|Egyptian Arabic|North Levantine|Hassaniya|Cypriot Arabic|Hijazi|Hejazi Arabic|Hejazi|Mesopotamian Arabic|Najdi|Yemeni Arabic|Saudi Arabic|Morrocan Arabic|North Mesopotamian Arabic|Omani Arabic"
luascript = "cd Lua_Modules/ && ./ar-TR.lua";
}
#
if(lang == "Mandarin")
{
generic_lang = "Chinese";
lang = "Mandarin|Central Mandarin|Jianghuai Mandarin|Northern Mandarin|West Mandarin|Wuhan|Xi[']an|Liuzhou|Chengdu|Xuzhou|Yangzhou|Ürümqi|Harbin|Simplified|Traditional|Chinese [(]Mandarin[)]|Chinese traditional[/]simplified|Chinese|Pinyin|Chinese [(]Traditional[)]|Chinese [(]Simplified[)]";
lang_qualifier = "|Central China|Jianghuai|Northern China|West China|Wuhan|Xi[']an|Liuzhou|Chengdu|Xuzhou|Yangzhou|Ürümqi|Harbin|"
iso = "zh|lzh|zho|chi|cmn|zh-tw|zh-cn|zhx-zho";
iso_qualifier = "|Literary Chinese|";
latin = 0;
exclude_lang = "Amoy|Bai|Cantonese|Changsha|Chaozhou|Dungan|Eastern Hokkien|Southern Hokkien|Eastern Min|Fuzhou|Gan|Guangzhou|Haikou|Hainanese|Hakka|Hangzhou|Hokkien|Hui|Jian[']ou|Jin|Jixi|Meixian|Min Bei|Min Dong|Min-nan|Min nan|Min Nan|Min-Nan|Nanchang|Nanning|Northern Hokkien|Northern Min|Northern Wu|Old Chinese|Pinghua|Shanghai|Shanghainese|Sichuanese|Southern Min|Southern Wu|Suzhou|Taiyuan|Taiwan|Taiwanese|Teochew|Tuhua Dong[']an|Wenzhou|Wu|Xiang|Xiamen|Yangzhou|Yue|Middle Chinese|Hoisanese|Taishanese"
}
#
if(lang == "Mandarin_nonested")
{
generic_lang = "Mandarin";
lang = "Mandarin";
generic_lang = "Mandarin";
enable_ttbc = 0;
latin = 0;
iso = "zh|cmn"
}
#
if(lang == "Persian")
{
iso = "fa";
exclude_lang = "Old Persian|Middle Persian|Eastern Persian";
latin = 0;
}
#
if(lang == "Kurdish")
{
iso = "ku|kmr|kur";
exclude_lang = "Sorani|Soranî|Central Kurdish|Southern Kurdish|Zazaki|Hewrami";
lang = "Kurmanji|Kurmancî|Kurdish";
generic_lang = "Kurdish";
latin = 0;
}
#
# Modern Greek
if(lang == "Greek")
{
iso = "el";
exclude_lang = "Ancient Greek|Ancient|Hebrew|Modern Romanization|Ancient Romanization|Mycenaean|Classical|Katharevousa|Katharevoussa|Pontic Greek|Koine|Roman|Cappadocian|Aeolic Greek|Pontic|Syriac|Griko|Medieval Greek|Byzantine Greek|Doric Greek|Laconian Greek|Doric|Laconian|Aeolic|Italiot Greek|Tsakonian";
lang = "Modern Greek|Modern|Greek";
generic_lang = "Greek";
latin = 0;
# remove transliterations (on user request) 
# rmtr = 1;
luascript = "cd Lua_Modules/ && ./el-TR.lua";
}
#
if(lang == "Indonesian")
{
iso = "id";
generic_lang = "Indonesian";
lang = "Indonesian|Standard Indonesian|Standard";
exclude_lang = "Acehnese|Balinese|Banjar|Banjarese|Buginese|Javanese|Kaili|Madurese|Makasar|Mandar|Minangkabau|Nias|Sasak|Sunda|Sundanese|Indonesian Bajau|Banda|Ende|Sikule|Simeulue";
}
#
if(lang == "Malay")
{
iso = "ms";
generic_lang = "Malay";
lang = "Rumi|Malay|Latin";
exclude_lang = "Malayalam|Malaysian Sign Language|Jawi|Arabic|Malayo-Polynesian|Jambi Malay|Kelantan-Pattani Malay|Malaynon";
}
#
if(lang == "Catalan")
{
iso = "ca";
lang = "Catalan|Valencian|Alguerese|Balearic";
iso_qualifier = "";
lang_qualifier = "|Valencian|Alguerese|Balearic";
}
#
if(lang == "Russian")
{
iso = "ru";
latin = 0;
luascript = "cd Lua_Modules/ && ./ru-TR.lua";
}
#
if(lang == "Serbo-Croatian")
{
iso = "sh|bs|hr|sr";
generic_lang = "Serbo-Croatian";
lang = "Serbo-Croatian|Serbian|Bosnian|Croatian|Roman|Latin|Cyrillic";
#exclude_lang="Cyrillic";
lang_qualifier = "|Serbian|Bosnian|Croatian|||";
# has to be configured as non-latin for now because many Cyrillic terms are not tagged 
latin = 0;
}
#
############################################
# END of user defined configuration section
############################################
#
#
if(LATIN == "n") latin = 0;
if(REMOVE_WIKILINKS == "y") remove_wikilinks = 1;
if((ISO != "")&&(iso == "")) iso = ISO;
if(GENERIC_LANG != "") generic_lang = GENERIC_LANG;
# set generic_lang to default: LANG 
if((LANG != "")&&(GENERIC_LANG == "")&&(generic_lang == "")) generic_lang = LANG;
if(ISO_QUALIFIER != "")  iso_qualifier = ISO_QUALIFIER;
if(LANG_QUALIFIER != "")  lang_qualifier = LANG_QUALIFIER;
if(TRANS_SEE == "y") enable_trans_see = 1;
if(EXCLUDE_LANG != "") exclude_lang = EXCLUDE_LANG;
if(ENABLE_IPA == "y") enable_ipa = 1;
if(REMOVE_TRANSLIT == "y") rmtr = 1;

# debug config output:
# print "lang="lang";iso="iso";generic_lang="generic_lang";exclude_lang="exclude_lang;
#
# write iso- and lang-qualifiers into array
n_iso=split(iso, iso_array, "|");
split(iso_qualifier, iso_qualifier_array, "|");
for(i=1; i<=n_iso; i++) {
if(iso_qualifier_array[i] == "") qualifier[iso_array[i]] = "";
else qualifier[iso_array[i]] = " [" iso_qualifier_array[i] "] ";
#print iso_array[i]" "qualifier[iso_array[i]];
}

n_lang=split(lang, lang_array, "|");
split(lang_qualifier, lang_qualifier_array, "|");
for(i=1; i<=n_lang; i++) {
if(lang_qualifier_array[i] == "") qualifier[lang_array[i]] = "";
else   qualifier[lang_array[i]] = " [" lang_qualifier_array[i] "] ";
#print lang_array[i]" "qualifier[lang_array[i]];
}

#
# initialization of variables used for parsing
#
# english = 0/1 outside/inside English section
english = 0; 
# trans = 0/1 outside/inside translations section
trans = 0;
# non-nested = 0/1 outside/inside non/nested translation line
non_nested = 0;
# gloss = gloss-string or empty
gloss = ""; 
# pos = part of speech
pos = ""; 
# title = pagetitle
title = "";
# inside nested section? 0/1
nestsect = 0;
# inside Pronunciation section? 0/1
pron = 0;
#
headline = 0;
# default IPA pronunciation
ipa1 = "";
# default IPA regexp
defipa="\\{\\{a\\|(US|GenAm).*\\{\\{IPA\\|";
# alternative IPA pronunciation
ipa2 = "";
# alternative IPA regexp
altipa = "\\{\\{IPA\\|";
#
oldLHS = ""; oldRHS = "";
# regexp matching start of nested section
if(enable_ttbc==1)
neststart = "^\\*[ ]*([[]*("generic_lang")|\\{\\{ttbc\\|("generic_lang")\\}\\}|\\{\\{ttbc\\|("iso")\\}\\}|\\{\\{trreq\\|("iso")\\}\\})"; 
if(enable_ttbc==0)
neststart = "^\\*[ ]*[[]*("generic_lang")"; 
# regexp matching translation lines to be excluded
exclude = "^$";
if(exclude_lang != "")
#exclude = "^[*:]*[\\x20]*[[]*("exclude_lang")[]]*[\\x20]*[:]";
exclude = "^[*:]*[ ]*[[]*("exclude_lang")";

# array containing POS header regexps and POS label
# indexed by frequency ('for var in array' gives arbitrary array sorting in awk)
PHR[1] = "Noun"; POSL[1] = "n";
PHR[2] = "Verb"; POSL[2] = "v";
PHR[3] = "(Adjective|Posesive[ ]adjective)"; POSL[3] = "adj";
PHR[4] = "(Adverb|Adverbial)"; POSL[4] = "adv";
PHR[5] = "Interjection"; POSL[5] = "interj";
PHR[6] = "Proper[ ]noun"; POSL[6] = "prop";
PHR[7] = "Phrase"; POSL[7] = "phrase";
PHR[8] = "Article"; POSL[8] = "article";
PHR[9] = "Preposition"; POSL[9] = "prep";
PHR[10] = "(Initialism|\\{\\{initialism)"; POSL[10] = "initialism";
PHR[11] = "(Number|Numeral)"; POSL[11] = "num";
PHR[12] = "Cardinal num(ber|eral)[ ]*[=]"; POSL[12] = "num";
PHR[13] = "Ordinal number"; POSL[13] = "num";
PHR[14] = "(Acronym|\\{\\{acronym)"; POSL[14] = "acronym";
PHR[15] = "(Abbreviation[= ]|\\{\\{abbreviation)"; POSL[15] = "abbr";
PHR[16] = "Determiner"; POSL[16] = "determiner";
PHR[17] = "Suffix"; POSL[17] = "suffix";
PHR[18] = "Pronoun"; POSL[18] = "pron";
PHR[19] = "Conjunction"; POSL[19] = "conj";
PHR[20] = "Proverb"; POSL[20] = "proverb";
PHR[21] = "Contraction"; POSL[21] = "contraction";
PHR[22] = "Particle"; POSL[22] = "particle";
PHR[23] = "Symbol"; POSL[23] = "symbol";
PHR[24] = "Prefix"; POSL[24] = "prefix";
PHR[25] = "Prepositional[ ]phrase"; POSL[25] = "prep";
PHR[26] = "Interfix"; POSL[26] = "interfix";
NPHR = 26;
# for(i=1;i<= NPHR;i=i+1)  { print i" "POSL[i];} exit(0);

# shortcuts for template names
shortcuts = "indeclinable|pf.|plural";
replacement = "indecl|pf|p";
n1 = split(shortcuts,sh_array,"|");
n2 = split(replacement,rep_array,"|");
if(n1 != n2) print "#WARNING: badly formatted template-name shortcut strings" >FIXME;
for(i=1; i<=n1; i++) { trep_text[sh_array[i]] = rep_array[i];
#print sh_array[i] " " trep_text[sh_array[i]];
}

# read trans-see from file in array
if(trans_see_file != "") {
while ((getline line < trans_see_file) > 0) {
	link = line;
	gsub(/^.*SEE[:][\ ][[]*/, "", link);
	gsub(/[]].*$/, "", link);
	if(link in trans_see_array)  trans_see_array[link] = trans_see_array[link] "\n" line;
	else trans_see_array[link] = line;
#	print link ": " trans_see_array[link];
}
}

}
# End BEGIN block
##############################################################


##############################################################
#
# Functions:
#
##############################################################

function replace_template(tpar, n_unnamed,     outp, i, j) {
# scans tpar and returns replacement string for the template
# tpar[0] is the template name
# tpar[1], ..., tpar[n_unnamed] are the unnamed parameters
# tpar["name1"], ...,  tpar["nameN"] are the named parameters with names name1, ..., nameN

# debug output
# for (j in tpar) print j, tpar[j];
MAXGENDERS = 5;

switch (tpar[0]) {

# templates to be removed
case /^(attention|rfc-tbot|inv|rfr|rfscript|rftranslit|NNBS|RL|LR|\,|jump|rfv|C|note|R[:].*|t-SOP|t[+]|p|f|n|m-p|f-p|n-p|webofscience|ISBN|topics|cite-web|biang|cite-book|a|enPR)$/:
return "";


# t-template
case "t":
outp = "";
if(1 in tpar) {

if(non_nested == 1) {
	if(tpar[1] in qualifier) outp = qualifier[tpar[1]] outp;
}

if(2 in tpar) {
if(tpar[2] ~ /\[\[/) outp = outp tpar[2];
else { 
	if("alt" in tpar)  outp = "[[" tpar[2] "|" tpar["alt"] "]]";
	else outp = "[[" tpar[2]  "]]";
	}
}

for(i=0;i<=MAXGENDERS;i++) 
	{ if(i+3 in tpar) outp = outp " {" tpar[i+3] "}";}

if((latin == 0)&&(rmtr == 0)) {
	if("tr" in tpar) outp = outp " /" tpar["tr"] "/";
	else
		if((luapipe == 1) && (luascript != "") && (2 in tpar)) {
			if(tpar[2] ~ /\[\[/) txt = linktotext(tpar[2]);
			else {	
				if("alt" in tpar) txt = tpar["alt"];
				else txt = tpar[2];
}
			print txt |& luascript; close(luascript, "to");
			luascript |& getline trsc; close(luascript);
			if((trsc != "nil")&&(trsc != ""))
				outp = outp " /" trsc "/";
}
}

if("lit" in tpar) outp = outp " (" tpar["lit"] ")";
re = "^(" iso ")$";
if(tpar[1] !~ re)
	print "#WARNING: wrong iso code: \"" tpar[1] "\" of t-template in entry: \"" title "\"  on line: \"" $0 "\"" > FIXME;

return mask_commas(outp);
}
else return "";

# qualifier
case /^(qualifier|i|italbrac|ib|qual|q|qf)$/:
outp = tpar[1];
for(i = 2; i in tpar; i++) outp = outp ", " tpar[i];
# remove wikilinks
gsub(/\[\[[^\]\[]*_BAR_/, "", outp);
gsub(/\[\[/, "", outp); gsub(/\]\]/, "", outp);
outp = "[" outp "]";
return mask_commas(outp);

# gloss-template
case /^(gloss|sense)$/:
outp = "(" tpar[1] ")";
return mask_commas(outp);

# trans-top: set gloss, then remove
case "trans-top":
gloss = tpar[1];
gsub(wlbar, "|", gloss); gsub(sob, "{", gloss); gsub(scb, "}", gloss);
return "";

# TODO: trans-top-also with only one argument
case "trans-top-also":
gloss = tpar[1] ", see also: " tpar[2];
for(i = 3; i in tpar; i++) gloss = gloss "; " tpar[i];
gsub(wlbar, "|", gloss); gsub(sob, "{", gloss); gsub(scb, "}", gloss);
return "";

# trans-see: set gloss and link, then remove
case "trans-see":
gloss = tpar[1];
gsub(/\[\[/, "", gloss); gsub(/\]\]/, "", gloss);
if(2 in tpar) link = tpar[2];
	else link = "[[" gloss "]]";
gsub(wlbar, "|", link); gsub(wlbar, "|", gloss); gsub(sob, "{", gloss); gsub(scb, "}", gloss); 
return "";

# the g-template
case "g":
outp = "";
for(i = 1; i in tpar; i++) outp = outp sob tpar[i] scb;
return outp;

# not used template
case "not used":
outp = "Not used in " generic_lang;
return outp;

# l-templates
# TODO: alt and tr parameters
case /^(l|l-self|link|m|mention|m-self|ll|l\/.*)$/:
if(tpar[2] ~ /\[\[/) outp = tpar[2];
	else outp = "[[" tpar[2] "]]";
return mask_commas(outp);

# templates to be replaced by [templatename]
case "no equivalent translation":
outp = "[no equivalent translation]";
return outp;

# templates to be replaced by "{templatename}"
case /^(indeclinable|pf.|plural)$/:
tpar[0] = trep_text[tpar[0]]; 
case /^(impf|dual)$/:
return sob tpar[0] scb;

# templates to be replaced by "templatename"
case /^(CE|BC|BCE)$/:
return  tpar[0];

# templates replaced by first unnamed parameter
case /^(unsupported|w|non-gloss definition|n-g|taxlink|IPAchar|W|upright)$/:
return tpar[1];

# templates replaced by 2nd unnamed parameter, e.g. lang-template
case "lang":
return tpar[2];

case "sup":
return "<sup>" tpar[1] " </sup>";

# pronunciation templates
case /^(IPA)$/:
if(headline == 1) {
	if(("lang" in tpar) && (tpar[1]!="")) ipa=tpar[1];
	else {if(tpar[2]!="") ipa=tpar[2];}
        gsub(/(\/|\[|\])/, "", ipa);
}
return "";

# obsolete or misplaced templates:
##################################

# lb-template: shouldn't be in a translations section
case /^(lb|label|lbl)$/:
outp = "[" tpar[2];
for(i = 3; i in tpar; i++) outp = outp ", " tpar[i];
outp = outp "]";
print "#WARNING: lb-template in entry: \"" title "\" on line: \"" $0 "\" detected" > FIXME;
return mask_commas(outp);

# obsolete term
case "term":
return "[[" tpar[1] "]]"

# Chinese and Japanese templates shouldn't be used in translation
case "zh-tsp":
return tpar[1] ", " tpar[2] ", /" tpar[3] "/";

case "zh-l":
return tpar[1];

case "ja-r":
return tpar[1];

case  "ja-l":
return "[[" tpar[1] "]] /" tpar[2] ", " tpar[3] "/";

# unknown templates are deleted with warning message so that we can fix this
default:
if(headline==0) print "#WARNING: deleted unknown template: {{" tpar[0] "}} in entry: \"" title "\" on line: \"" $0 "\"" > FIXME;
return "";
}
}

#####################################
function parse_templates(input,         i, j, k, ta, sa, nt, ts, na, targs, n2, a2, tpar, rep, outp)
{
# parses string for templates 
# and calls replace_template() for each template found
# then returns a replacement string
# THIS FUNCTION HAS TO BE CALLED MULTIPLE TIMES FOR STRINGS WITH NESTED TEMPLATES

# escape code to: 
#replace bars inside wikilinks with wlbar
wlbar="_BAR_";
# replace single braces
sob="_LeftBrace_";
scb="_RightBrace_";

input = gensub(/([^\{])(\{)([^\{])/, "\\1" sob "\\3", "g", input);
input = gensub(/([^\}])(\})([^\}])/, "\\1" scb "\\3", "g", input);
gsub(/\{\{[=]\}\}/, "\\&equals;", input);


# is this necessary?
delete ta; delete sa;

# split input string into templates (ta[1, ..., n]) and non-template strings (sa[0, ..., n])
nt = patsplit(input, ta, /\{\{[^}{]*\}\}/, sa);

output = "";
for(i=1; i<=nt; i=i+1) {
# replace_template(ta[i]);
	ts = ta[i]
# replace bars inside wikilinks with wbar
	ts = gensub(/(\[\[[^\]]*)(\|)([^\]]*\]\])/, "\\1" wlbar "\\3", "g", ts); 

# split template args into array targs	
	sub(/\{\{/, "", ts); sub(/\}\}/, "", ts);
	na = split(ts, targs, "|");

	k = 0; delete tpar;
	for(j = 1; j <= na; j = j+1) {
		n2 = split(targs[j], a2, "=");
# prevent uninitialized  a2[1] for empty template argument targs[j]
		if(n2 == 0)  a2[1] = "";
		if(n2 <= 1) {tpar[k] = a2[1]; k=k+1;}
			else tpar[a2[1]] = a2[2];
		}

# debug output
# for (test in tpar) print test, tpar[test];
# now call replace_template function which returns a replacement string for the template
	rep = replace_template(tpar, k-1);
	ta[i] = rep;
	}

outp = "";
if(0 in sa) outp = sa[0]; 
for(i = 1; i <= nt; i = i+1) { outp = outp ta[i]; if(i in sa) outp = outp sa[i];}
return outp;
}

#####################################
function printout(out) {
# does special formatting before output

# escaped bars and curly braces
gsub(wlbar, "|", out); gsub(sob, "{", out); gsub(scb, "}", out); 

# these are from formatting errors:
gsub("[[]]", "", out);
gsub("{}", "", out);

# remove links to sections			
regexp = "\\[\\[\\#("lang")\\|";
gsub(regexp, "[[", out);
regexp = "#("lang")\\|";
gsub(regexp, "|", out);

# convert common gender "{c}" to "{m} {f}" for languages de, es, fr, it, pt
if((iso=="de")||(iso=="es")||(iso=="fr")||(iso=="it")||(iso=="pt")) {
	gsub(/\{\{c\}\}/,"{mf}",out);
	gsub(/\{c\}/,"{mf}",out);
}

# remove empty wikilinks
gsub(/\[\[[ ]*\]\]/, "", out);

# remove <\/text>, (might be there at the end of page (XML-code)			
gsub(/<\/text>/,"",out);

# convert special XML formatting like &lt; to text
gsub(/&amp;/,"\\&",out);
gsub(/&lt;/,"<",out);
gsub(/&gt;/,">",out);
gsub(/&quot;/,"\"",out);
gsub(/&nbsp;/, " ", out);
gsub(/&hellip;/, "...", out);
gsub(/&quot;/, "\"", out);
gsub(/&[mn]dash;/, "-", out);
gsub(/&thinsp;/, "", out);
gsub(/&minus;/, "-", out);
gsub(/&equals;/, "=", out);
gsub(/&#39;/, "'", out);
gsub(/&#61;/, "=", out);
gsub(/&frac12;/, "½", out);


# convert masked commas back
gsub(/&comma;/, ",", out);

# NOTE: these must be done after converting '&lt;' -> '<'  and '&gt;' -> '>'
# remove <ref ... \>
gsub(/<ref[^>]*\/>/,"",out);

# remove <ref [name=....]> blabla </ref> OK?
gsub(/<ref[^>]*>.*<\/ref>/,"",out);

# <ref at end of line:
gsub(/<ref[^>]*>.*$/,"",out);

# remove one-line <!-- commented text -->
gsub(/<!--[^>]*-->/,"",out); 

# remove extra spaces
gsub(/[\ ]+/, " ", out);

# remove remaining "<!--" (will prevent display of wikifile)
gsub(/<!--/,"", out);

# remove unicode left-to-right, right-to-right etc.: U+200E, U+200F
gsub(/\xE2\x80(\x8E|\x8F)/, "", out);

if(remove_wikilinks==1) {
# wikilinks and italicizing, bolding
	out = gensub(/([[][[])([^]|]*\|)([^]]*)([]][]])/ , "\\3", "g", out);
	out = gensub(/([[][[])([^]]*)([]][]])/ , "\\2", "g", out);
	gsub(/['][']+/, "", out);

# <sub> and <sup>
	gsub(/<sup>/, "^", out);  gsub(/<\/sup>/, "", out);
	gsub(/<sub>/, "", out);  gsub(/<\/sub>/, "", out);
			 
# <nowiki> 			
	gsub(/<nowiki>/, "", out); gsub(/<\/nowiki>/, "", out);	
}

# force LR-switch for some characters
if((generic_lang == "Arabic")&&(remove_wikilinks == 0)) {
	gsub(/[]][ ]*[/]3/, "] {{LR}}/3", out);
}

# remove diacritics for Cyrillic, Arabic and Hebrew script:
out = remove_diacritics(generic_lang, out);

print out;
}

########################
function linktotext(text) {
gsub(/_BAR_/, "|", text);
text = gensub(/([[][[])([^]|]*\|)([^]]*)([]][]])/ , "\\3", "g", text);
text = gensub(/([[][[])([^]]*)([]][]])/ , "\\2", "g", text);
return text;
}

####################
function set_LHS() {
LHS = "[[" title "]] "; 
if(pos == "") {
	pos = "?"; 
	print "#WARNING: unknown POS on page:\""title"\"" > FIXME;
}
LHS = LHS "{" pos "} ";
if(enable_ipa==1) {
	if(ipa1!="") { LHS = LHS "/" ipa1 "/ "; ipa1=""; ipa2="";}
	if(ipa2!="") { LHS = LHS "/" ipa2 "/ "; ipa1=""; ipa2="";}
}	
if (gloss != "") LHS = LHS "(" gloss ") ";

return LHS;	
}

####################################
function print_trans_see(oldtitle) {
# printout matching trans-see links if enabled
if(trans_see_file != "") {	
	if(oldtitle in trans_see_array) printout(trans_see_array[oldtitle]);
	delete trans_see_array[oldtitle];
}
}

#####################################################
function remove_diacritics(generic_lang, text) {
# info about diacritics/vocalization removal is located in Module:languages/data/2, data/3
# languages/data gives the unicode codepoints which have to be converted to utf8-hex

switch (generic_lang) {

case "Russian":
text = gensub(/([аеёиоуыэюяАЕЁИОУЫЭЮЯ])(\xCC\x81|\xCC\x80)/, "\\1",  "g", text);
return text;

case "Bulgarian":
gsub(/(\xCC\x81|\xCC\x80)/, "", text);
gsub(/Ѐ/, "Е", text);
gsub(/ѐ/, "е", text);
gsub(/Ѝ/, "И", text);
gsub(/ѝ/, "и", text);
return text;

case "Macedonian":
gsub(/\xCC\x81/, "", text);
return text;

case "Serbo-Croatian":
gsub(/(\xCC\x81|\xCC\x80|\xCC\x8F|\xCC\x91|\xCC\x84|\xCC\x83)/, "", text);
gsub(/ȀÀȂÁĀÃ/, "A", text);
gsub(/ȁàȃáāã/, "a", text);
gsub(/ȄÈȆÉĒẼ/, "E", text);
gsub(/ȅèȇéēẽ/, "e", text);
gsub(/ȈÌȊÍĪĨ/, "I", text);
gsub(/ȉìȋíīĩ/, "i", text);
gsub(/ȌÒȎÓŌÕ/, "O", text);
gsub(/ȍòȏóōõ/, "o", text);
gsub(/ȐȒŔ/, "R", text);
gsub(/ȑȓŕ/, "r", text);
gsub(/ȔÙȖÚŪŨ/, "U", text);
gsub(/ȕùȗúūũ/, "u", text);
gsub(/Ѐ/, "Е", text);
gsub(/ѐ/, "е", text);
gsub(/ӢЍ/, "И", text);
gsub(/Ӯ/, "У", text);
gsub(/ӯ/, "у", text);
return text;

case "Arabic":
gsub(/\xD9\xB1/, "\xD8\xA7", text);
gsub(/\xD9(\x8B|\x8C|\x8D|\x8E|\x8F|\x90|\x91|\x92|\xB0|\x80)/, "", text);
return text;

case "Persian":
gsub(/\xD9(\x8E|\x8F|\x90|\x91|\x92)/, "", text);
return text;

case "Hebrew":
gsub(/\xD6(\x91|\x92|\x93|\x94|\x95|\x96|\x97\x98|\x99|\x9A|\x9B\x9C|\x9D|\x9E|\x9F|\xA0|\xA1|\xA2|\xA3|\xA4|\xA5|\xA6|\xA7|\xA8|\xA9|\xAA|\xAB|\xAC|\xAD|\xAE|\xAF|\xB0|\xB1|\xB2|\xB3|\xB4|\xB5|\xB6|\xB7|\xB8|\xB9|\xBA|\xBB|\xBC|\xBD|\xBF)/, "", text)
gsub(/\xD7(\x80|\x81|\x82|\x83|\x84|\x85|\x86|\x87)/, "", text)
return text;

default:
return text;
}
}

############################################################
function mask_commas(text) {
# replace commas and semicolon inside trans-, qualifier- l- and gloss-templates
# so that commas and semicolons separating translation	con be substituted by
# a custom character	
gsub(/[,]/, "\\&comma;", text);
return text;
}

############################################################
#
# Main Program Blocks
#
############################################################

# get page title
/[<]title[>]/ {

gsub(/^[^<]*[<]title[>]/, ""); gsub(/[<][/]title[>].*$/, ""); 
title=$0;
english=0; trans=0; gloss=""; pos=""; nestsect=0; text=0;

if(index(title,"/translations") != 0) {
english = 1;
#remove translations sub-page from title
gsub(/\/translations/, "", title);
}

if(index(title,"Wiktionary:") != 0) title="";
if(index(title,"Template:") != 0) title="";
if(index(title,"Appendix:") != 0) title="";
if(index(title,"User:") != 0) title="";
if(index(title,"Help:") != 0) title="";
if(index(title,"MediaWiki:") != 0) title="";
if(index(title,"Thesaurus:") != 0) title="";
}

# discard Wiktionary, Template and Appendix namespaces
{if(title == "") next;}

# discard non-useful lines (speedup and false "trans-see" lines from comment lines)
#/<comment>|<\/?page>|<timestamp>|<id>|<\/?contributor>|<\/?revision>|<username>|<minor \/>/  {next;}
/[<]text/ {text =1;}
{if(text == 0) next;}

# detect English language section
/[=][=][ ]*English[ ]*[=][=]/ { 
english = 1;
trans = 0; gloss = ""; pos = ""; nestsect = 0; pron = 0; ipa1 = ""; ipa2 = "";
next;
}

# skip transligual sections
/[=][=][ ]*Translingual[ ]*[=][=]/ {english == 0; next;}

# detect non-English language section
# English and Transligual are first, so we can skip the remainig entries
/^[\ ]*[=][=][^=]+/ {title = ""; next;}

# language and page title detection done; skip all lines if not inside an English section
{if(english == 0) next;}

#################################################
# Now inside English section
#################################################

# detect pronunciation section
/[=][=][=][ ]*Pronunciation/ {pron = 1; ipa1 = ""; ipa2 = "";}
# determine ipa1 and ipa2 pronunciation strings
$0 ~ defipa {
if((pron == 1)&&(ipa1 == "")) { 
	gsub(/\{\{[!]\}\}/, wlbar, $0);
        ipa = "";       
        headline=1;
# parse gender in headline-template via replace_template function
        HD = $0;
        HD = parse_templates(HD);
# do we have nested headlines? would require parsing twice:
        parse_templates(HD);                            
        ipa1=ipa;
        headline=0;
#	gsub(/\|lang\=en/, "", $0);
#	ipa1 = gensub(/(.*\{\{IPA\|[\/\[]*)([^}\|\/]*)([\/\]]*.*)/, "\\2", "g", $0); 
# print "def "title" "ipa1 >>"IPA.txt";
next;
}
}

$0 ~ altipa {
if((pron == 1)&&(ipa2 == "")) {
	gsub(/\{\{[!]\}\}/, wlbar, $0);
#	gsub(/\|lang\=en/, "", $0);	
#	ipa2=gensub(/(.*\{\{IPA\|[\/\[]*)([^}\|\/]*)([\/\]]*.*)/, "\\2", "g", $0); 
        ipa = "";       
        headline=1;
# parse gender in headline-template via replace_template function
        HD = $0;
        HD = parse_templates(HD);
# do we have nested headlines? would require parsing twice:
        parse_templates(HD);                            
        ipa2=ipa;
        headline=0;
# print "alt "title" "ipa2 >>"IPA.txt";
next;
}
}

# get part of speech (POS)
/[=][=][=]/ {
for(i=1;i<= NPHR;i=i+1) {
	if($0 ~ PHR[i]) { 
		pos=POSL[i]; trans=0; gloss = ""; next;
} 
}
}

# detect end of Translations section
/[=][=]/ {
trans = 0; nestsect = 0;
}

# detect start of Translations section
/[=][=][=][=][ ]*Translations/ {
if(english == 1) {trans = 1; gloss = ""; nestsect = 0;}
next;
}

/[=][=][ ]*Translations/ {
if(english == 1) {trans = 1; gloss = ""; nestsect = 0;
	print "WARNING: non-compliant translation section level (<4) on page: " title >FIXME;}
next;
}

######################################################################
# Now we can skip all lines if we're not inside a translations section
{if(trans == 0)  next;}


# detect start of Checktrans section
/\{\{checktrans/ {gloss = ""; nestsect = 0;}

# get gloss from trans-top
/\{\{trans-top\||\{\{trans-top-also\|/ {
gloss = "";
TR = parse_templates($0);
TR = parse_templates(TR);
gsub(/\([1-9]\)/, "", gloss);
nestsect = 0;
}

# handle {{trans-see||}} links
/\{\{trans-see\|/ {
if(enable_trans_see == 1) {
	gloss = "";	
	TR = parse_templates($0);
	TR = parse_templates(TR);
# print "TRANS-SEE: "$0 " " gloss " " link;		
	LHS = set_LHS();
	if(index(link,"[[") == 0)	
		outp = LHS " SEE: [["link"]] " Sep;
	else outp = LHS " SEE: "link" " Sep;

	printout(outp);
}
gloss=""; nestsect = 0;
}

# detect nested section
/^[*][^*:]|\{\{ttbc|\{\{trans-|\{\{trreq|^[[]/ {nestsect = 0;}
$0 ~ neststart {nestsect = 1;}

# skip lines matching exclude
$0 ~ exclude {next;}

# skip {{trreq| ... lines
/\{\{trreq\|/  {next;}

# skip {{t-needed| ... lines
/\{\{t-needed\|/  {next;}


###############################
# parse a valid translation line:

/^[*]/ {
if((trans == 1)&&(nestsect == 1)) {
	non_nested=0;
# set LHS
	LHS = set_LHS();
	TR = $0;

# conversion of obsolete/redirected/equivalent/recently changed templates
# the translation templates
	gsub(/\{\{(t[-]simple|apdx[-]t|t[-]SOP|t[+]|t[-]|tø|t0|t-check|t[+]check)\|/, "{{t|", TR);

# add qualifiers via qualifier[] and remove languages from start of translation line
# non-nested translation line: qualifier handled by replace_template() 
	regexp = "^\\*[ ]*[[]*"generic_lang"[ ]*[]]*[:]";
	if(TR ~ regexp) non_nested = 1;
# nested translation line
	for(i=1;i<=n_lang;i++) {
		regexp="^[*:]*[ ]*[[]*"lang_array[i]"[]]*[ :]*|^[*:]*[ ]*\\{\\{ttbc\\|"lang_array[i]"\\}\\}[ :]*";
		gsub(regexp, qualifier[lang_array[i]], TR);
}
	for(i=1;i<=n_iso;i++) {
		regexp="^[*:]*[ ]*\\{\\{ttbc\\|"iso_array[i]"\\}\\}[ :]*";
		gsub(regexp, qualifier[iso_array[i]], TR);
}
# remove remaining "^** " from nested translation lines
	sub(/^[*:]*[ ]*/, "" ,TR);

# convert templates using parse_templates():
#############################################

# convert {{=}} {{!}} 
	gsub(/\{\{[=]\}\}/, "\\&equals;", TR);
	gsub(/\{\{[!]\}\}/, wlbar, TR);

# template nesting level: MAXNESTING == 1 <=> no nesting
	MAXNESTING = 3;
	for(i = 1; i <= MAXNESTING ; i = i + 1) {
		TR = parse_templates(TR);
		if(TR !~ /\{\{/) break;
}

	if(TR ~ /\{\{/) {
		print "#WARNING: at entry: \"" title "\": skipping badly formatted input line: \"" $0 "\" or maybe to much template nesting, try to increase the \"MAXNESTING\" variable" > FIXME;
		next;
}

# remove (1) and the like (old disambiguation)
	gsub(/\([0-9 ,;-]*\)/, "", TR);

# replace separating commas with semicolons
if(TransSep != "") gsub(/[,]/, TransSep, TR);

# joining of output line
	if(LHS == oldLHS) {
		if(TR != "") {
			if(oldRHS != "") oldRHS = oldRHS TransLineSep TR;
			if(oldRHS == "") oldRHS = TR;
}
}

if(LHS != oldLHS) {
	if(oldRHS != "") {
		outp = oldLHS Sep " " oldRHS; 
		print_trans_see(oldtitle);
		printout(outp);
}
	oldLHS = LHS;
	oldRHS = TR;
	oldtitle = title;
}

# end trans=1
}
next;
# end of detected translations	line
}

# cleanup after end of input
END {
if(oldRHS != "") {
		outp = oldLHS Sep " " oldRHS;
		print_trans_see(oldtitle);
		printout(outp);}

if(luapipe == 1) close(luascript, "from");
}