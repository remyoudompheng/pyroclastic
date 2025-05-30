import argparse
import logging

from . import gpu
from . import sieve
from . import linalg


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("-v", "--verbose", action="store_true")
    argp.add_argument("--check", action="store_true", help="Verify relations")
    argp.add_argument(
        "-j",
        metavar="THREADS",
        type=int,
        default=2,
        help="Number of CPU threads (and parallel GPU jobs)",
    )
    argp.add_argument(
        "--ngpu",
        metavar="GPUS",
        type=int,
        default=1,
        help="Number of GPUs (usually a divisor of THREADS)",
    )
    argp.add_argument("N", type=int)
    argp.add_argument("OUTDIR")
    args = argp.parse_args()
    args.DATADIR = args.OUTDIR
    args.bench = False

    logging.getLogger().setLevel(logging.INFO)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.info(f"Running on device {gpu.device_name()}")
    sieve.main_impl(args)
    linalg.main_impl(args)


if __name__ == "__main__":
    main()
