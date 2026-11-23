# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: 2025-2026 Rémy O

import argparse
import logging
import tempfile

from . import gpu
from . import groupstruct
from . import linalg
from . import logs
from . import sieve


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("-v", "--verbose", action="store_true")
    argp.add_argument("--otlp", action="store_true")

    sieve_args = argp.add_argument_group("Sieve options")
    sieve_args.add_argument("--check", action="store_true", help="Verify relations")
    sieve_args.add_argument("--siever2", action="store_true", help="Use Siever2")

    linalg_args = argp.add_argument_group("Linear algebra options")
    linalg_args.add_argument("--deterministic", action="store_true")
    linalg_args.add_argument(
        "--ncpu",
        type=int,
        default=None,
        help="Number of CPU threads for (block) Wiedemann",
    )
    linalg_args.add_argument(
        "--blockm", type=int, default=2, help="Block size for Block Wiedemann"
    )
    linalg_args.add_argument(
        "--skip-structure", action="store_true", help="Don't compute group structure"
    )

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
    if not args.skip_structure:
        groupstruct.main_impl(args)


if __name__ == "__main__":
    main()
