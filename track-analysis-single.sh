f=$1

echo $f, $(grep "tube_r:" $f), $(grep "Done Flux" $f | wc -l), $(grep "Analysis Complete" $f | wc -l), $(grep "Error:" $f)
