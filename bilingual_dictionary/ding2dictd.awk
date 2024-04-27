BEGIN {FS=" :: "; print "#";}
/^[#]/ {
	print " "$0;
	header=1; next;}
/^[^#]/ {
if(header==1) {print "#\n#"; header=0;}
indx=$0;
# convert {...} -> <...> (curly brackets are interpreted as links by some clients)
gsub(/\{[^\{\}]*\}/, "<&>"); gsub(/<\{/, "<"); gsub(/\}>/, ">");
#rm See:...
gsub(/SEE[\:].*/, "", indx);
#rm {...}, [...] (...)
gsub(/\{[^}]*}/, "", indx);
gsub(/\([^)(]*)/, "", indx);
gsub(/\([^)(]*)/, "", indx);
gsub(/\[[^][]*]/, "", indx);
# convert ,;|/ -> ::
gsub(/[;,/]/, "::", indx);
# rm extraspace after and before ::
gsub(/[ ]*[\:][\:][ ]*/, "::", indx)
# rm multiple ::
gsub(/[\:][\:]+/, "::", indx);
# rm space at end of line
gsub(/[ ]*$/, "", indx);
gsub(/[ ]*[\:][\:]$/, "", indx);
# rm "::" at the end of trans-see
gsub(/[ ]*[\:]+[ ]*$/, "", $1);
# add dictd links {...} to trans-see (needs gawk)
# undocumented feature does not work with all clients
$1=gensub(/(.*SEE[:][\ ])(.*)/, "\\1{\\2}", "g", $1);
# printout		}
print indx;
print " "$1"\n\t"$2;
}