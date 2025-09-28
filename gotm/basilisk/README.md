Data retrieved from GOTM repo `/code-5.2.1/cases/ows_papa/`

List of files used for running the code (profile and baseline fluxes):
- sprof_papa_hourly.dat -> sprof.dat
- tprof_papa_hourly.dat -> tprof.dat
- swr_papa.dat -> swr.dat
- swr_papa.dat -> lwr.dat
- heat_flux_papa.dat -> heatflux.dat
- momentum_flux_papa.dat -> momentumflux.dat

List of meteo variables used to generate fluxes:
- sst_hourly.dat
- u10.dat
- airt.dat
- airp.dat
- hum.dat

Compile and run with the following:

```
# Get a computational node
srun --pty /bin/bash

# Go into singularity
# Or singgotm for alias
singularity exec /scratch/work/public/singularity/ubuntu-22.04.4.sif /bin/bash

# Recompile if made any changes (notice the build-2)
# Code using kpp and k-epsilon are compiled with different flag and named differently
qcc -disable-dimensions -autolink -O2 -DMTRACE=3 -g -Wall -pipe -D_FORTIFY_SOURCE=2 -DKPP=1 -o kpp_dt ows_papa_dt.c -lm -L/home/jw8736/code-5.2.1/build-2
qcc -disable-dimensions -autolink -O2 -DMTRACE=3 -g -Wall -pipe -D_FORTIFY_SOURCE=2 -DKPP=0 -o kepsilon_dt ows_papa_dt.c -lm -L/home/jw8736/code-5.2.1/build-2

# Run a job for an ensemble of cases with different forcing
# slurm.sh creates an array of jobs with different years saved to separate folders
# which calls job.sh that does monthly restart and multiple forcing files 
sbatch slurm.sh 
```