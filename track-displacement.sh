fnames=$(find . -name "radial_displacement.o${1}.*")

for f in $fnames
  do
    echo $f, $(grep "Starting Tube " $f | tail -n 1), $(grep "Starting height" $f | tail -n 1), $(grep "Completed tube" $f | wc -l), $(grep "Completed height" $f | wc -l), $(grep "Error:" $f)
  done
