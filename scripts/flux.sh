#!/bin/bash

# Need to run the following in command line before this script
# singexec
# load_env
# python fluxgen.py --help for usage

ENSEM=20
IPATH=/home/jw8736/code-5.2.1/cases/ows_papa/
OPATH=/home/jw8736/test-gotm/ensem/

# python gen_flux.py --sd=2012-03-21 --ed=2012-05-20 --corrtime=60 --dt=3 --ensem=$ENSEM --flux=heat -i=$IPATH -o=$OPATH
# python gen_flux.py --sd=2012-03-21 --ed=2012-05-20 --corrtime=15 --dt=3 --ensem=$ENSEM --flux=momentum -i=$IPATH -o=$OPATH

# python gen_flux.py --sd=2012-05-21 --ed=2012-08-20 --corrtime=60 --dt=3 --ensem=$ENSEM --flux=heat -i=$IPATH -o=$OPATH
# python gen_flux.py --sd=2012-05-21 --ed=2012-08-20 --corrtime=15 --dt=3 --ensem=$ENSEM --flux=momentum -i=$IPATH -o=$OPATH

# python gen_flux.py --sd=2012-08-21 --ed=2013-01-10 --corrtime=60 --dt=3 --ensem=$ENSEM --flux=heat -i=$IPATH -o=$OPATH
# python gen_flux.py --sd=2012-08-21 --ed=2013-01-10 --corrtime=15 --dt=3 --ensem=$ENSEM --flux=momentum -i=$IPATH -o=$OPATH

# python gen_flux.py --sd=2013-01-11 --ed=2013-03-20 --corrtime=60 --dt=3 --ensem=$ENSEM --flux=heat -i=$IPATH -o=$OPATH
# python gen_flux.py --sd=2013-01-11 --ed=2013-03-20 --corrtime=15 --dt=3 --ensem=$ENSEM --flux=momentum -i=$IPATH -o=$OPATH

# Generate the full time 
python gen_flux.py --sd=2012-01-01 --ed=2020-01-01 --corrtime=60 --dt=3 --ensem=$ENSEM --flux=heat -i=$IPATH -o=$OPATH
python gen_flux.py --sd=2012-01-01 --ed=2020-01-01 --corrtime=15 --dt=3 --ensem=$ENSEM --flux=momentum -i=$IPATH -o=$OPATH