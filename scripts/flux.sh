#!/bin/bash

# Need to run the following in command line before this script
# singexec
# load_env
# python fluxgen.py --help for usage

# TODO: trouble shoot - somehow the generation of flux is not writing when multiple years. Need to trouble shoot...

ENSEM=20
IPATH=/home/jw8736/code-5.2.1/cases/ows_papa/
OPATH=/scratch/jw8736/gotm/ensem/

# python gen_flux.py --sd=2012-03-21 --ed=2012-05-20 --corrtime=60 --dt=3 --ensem=$ENSEM --flux=heat -i=$IPATH -o=$OPATH
# python gen_flux.py --sd=2012-03-21 --ed=2012-05-20 --corrtime=15 --dt=3 --ensem=$ENSEM --flux=momentum -i=$IPATH -o=$OPATH

# python gen_flux.py --sd=2012-05-21 --ed=2012-08-20 --corrtime=60 --dt=3 --ensem=$ENSEM --flux=heat -i=$IPATH -o=$OPATH
# python gen_flux.py --sd=2012-05-21 --ed=2012-08-20 --corrtime=15 --dt=3 --ensem=$ENSEM --flux=momentum -i=$IPATH -o=$OPATH

# python gen_flux.py --sd=2012-08-21 --ed=2013-01-10 --corrtime=60 --dt=3 --ensem=$ENSEM --flux=heat -i=$IPATH -o=$OPATH
# python gen_flux.py --sd=2012-08-21 --ed=2013-01-10 --corrtime=15 --dt=3 --ensem=$ENSEM --flux=momentum -i=$IPATH -o=$OPATH

# python gen_flux.py --sd=2013-01-11 --ed=2013-03-20 --corrtime=60 --dt=3 --ensem=$ENSEM --flux=heat -i=$IPATH -o=$OPATH
# python gen_flux.py --sd=2013-01-11 --ed=2013-03-20 --corrtime=15 --dt=3 --ensem=$ENSEM --flux=momentum -i=$IPATH -o=$OPATH

# Generate the full time 
python gen_flux.py --sd=2011-01-01 --ed=2020-01-01 --corrtime=60 --dt=1 --ensem=$ENSEM --flux=heat -i=$IPATH -o=$OPATH
# python gen_flux.py --sd=2012-01-01 --ed=2020-01-01 --corrtime=15 --dt=1 --ensem=$ENSEM --flux=momentum -i=$IPATH -o=$OPATH
