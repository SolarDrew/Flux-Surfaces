dnames=$(find /fastdata/sm1ajl/Flux-Surfaces/data/ -type d -name m-1_p60*)

for d in $dnames
  do
  echo $d
  for r in $(ls $d)
    do
      echo $r: $(ls $d/$r/*_h*_distance.npy | wc -l)
    done
  done
