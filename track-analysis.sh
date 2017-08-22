fnames=$(find . -name "driver_mx_analysis.o${1}.*" | sort -n)

echo -e 'Filename \t\t\t\t| Mode, tube \t| Done Flux \t| Analysis Complete \t| Error'
for f in $fnames
  do
    echo -e $f '\t|' $(grep ", r" $f) '\t|' $(grep "Done Flux" $f | wc -l) '\t|' $(grep "Analysis Complete" $f | wc -l) '\t|' $(grep "Error:" $f)
  done
