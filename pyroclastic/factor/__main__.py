import argparse
import logging
import tempfile

from pyroclastic import gpu
from pyroclastic import logs
from . import sieve
from . import linalg


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("-v", "--verbose", action="store_true")
    argp.add_argument("--otlp", action="store_true")

    sieve_args = argp.add_argument_group("Sieve options")
    sieve_args.add_argument("--dev", help="List of GPU device ids (example: 0,1,2)")

    linalg_args = argp.add_argument_group("Linear algebra options")
    linalg_args.add_argument("--deterministic", action="store_true")

    argp.add_argument(
        "-j",
        metavar="THREADS",
        type=int,
        default=0,
        help="Number of CPU threads (and parallel GPU jobs)",
    )
    argp.add_argument("N", type=int)
    argp.add_argument("OUTDIR", nargs="?")
    args = argp.parse_args()
    args.bench = False

    level = logging.DEBUG if args.verbose else logging.INFO
    logs.setup(level)
    if args.otlp:
        logs.setup_otlp()

    gpu.banner()

    if args.OUTDIR is None:
        logging.getLogger("main").info("Creating temporary directory for results")
        with tempfile.TemporaryDirectory(prefix="pyroclastic") as tmpdir:
            args.OUTDIR = tmpdir
            args.DATADIR = args.OUTDIR
            main_impl(args)
    else:
        args.DATADIR = args.OUTDIR
        main_impl(args)


def main_impl(args):
    sieve.main_impl(args)
    linalg.main_impl(args)


if __name__ == "__main__":
    main()
