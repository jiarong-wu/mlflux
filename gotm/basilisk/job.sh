#!/bin/bash
method=${1} # method = {kpp,kepsilon}
dt=${2} # time step in minutes
year=${3} # year 

OUTFOLDER=out_${method}_dt${dt}_${year}
EXE=${method}_dt
mkdir ${OUTFOLDER}
cp ${EXE} ${OUTFOLDER}
cp sprof.dat ${OUTFOLDER}
cp tprof.dat ${OUTFOLDER}
cp swr.dat ${OUTFOLDER}
cp momentumflux.dat ${OUTFOLDER} # for perturbed heat flux momentum is kept the same
cd ${OUTFOLDER}

for i in {1..20}; do
       cp ../heatflux_ann_ensem$i.dat ./heatflux.dat
       for month in {1..12}; do
              echo "Year ${year} Month ${month} Ensem member $i"
              ./${EXE} $year $month 1 $dt > out_${month}_ensem$i
       done
done

cp ../heatflux_ann_mean.dat ./heatflux.dat
for month in {1..12}; do
       ./${EXE} $year $month 1 $dt > out_ann_mean_${month}
done

cp ../heatflux_ann_ensem_mean.dat ./heatflux.dat
for month in {1..12}; do
       ./${EXE} $year $month 1 $dt > out_ensem_mean_${month}
done

cp ../heatflux_bulk.dat ./heatflux.dat
for month in {1..12}; do
       ./${EXE} $year $month 1 $dt > out_bulk_${month}
done

rm heatflux.dat
