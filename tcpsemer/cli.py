import argparse
import dataclasses
import json
import logging
import sys

from .core import DEFAULT_MODEL, combine, tcpsemer


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="tcpsemer", add_help=False,
        description="Time-constrained minimum-permutation semantic error rate.")
    p.add_argument("-r", "--reference", required=True, nargs="+",
                   help="Reference file(s) (SegLST JSON, STM, ...)")
    p.add_argument("-h", "--hypothesis", required=True, nargs="+",
                   help="Hypothesis file(s) (SegLST JSON, STM, ...)")
    p.add_argument("--help", action="help", help="Show this help and exit")
    p.add_argument("--collar", type=float, default=5.0,
                   help="Collar in seconds (default: 5)")
    p.add_argument("--regex", default=None, help="Only score sessions matching this regex")
    p.add_argument("--uem", default=None, nargs="+", help="UEM file(s)")
    p.add_argument("--normalizer", default=None,
                   help='Text normaliser as in meeteval-wer, e.g. "chime8"')
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"sentence-transformers model (default: {DEFAULT_MODEL})")
    p.add_argument("--device", default=None, help="Device for the embedding model")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-gap", type=float, default=2.0,
                   help="Reference gap (s) that starts a new utterance (default: 2)")
    p.add_argument("--backend", default="auto", choices=["auto", "rust", "python"],
                   help="Alignment backend (default: Rust if available)")
    p.add_argument("--per-reco-out", default=None,
                   help="Write per-session results to this JSON file")
    p.add_argument("--segments-out", default=None,
                   help="Write per-utterance distances to this JSON-lines file")
    p.add_argument("--average-out", default=None,
                   help="Write the overall result to this JSON file")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print per-session results")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    unwrap = lambda x: x[0] if x is not None and len(x) == 1 else x
    res = tcpsemer(unwrap(a.reference), unwrap(a.hypothesis), collar=a.collar,
                   regex=a.regex, uem=unwrap(a.uem), normalizer=a.normalizer,
                   model=a.model, device=a.device, batch_size=a.batch_size,
                   max_gap=a.max_gap, return_segments=a.segments_out is not None,
                   backend=a.backend)
    total = combine(res)

    row = lambda r: {"error_rate": r.error_rate, "errors": r.errors, "length": r.length}
    if a.verbose:
        for sess, r in sorted(res.items()):
            print(f"{sess}\ttcpSemER={r.error_rate:.2%}\terrors={r.errors:.2f}\tlength={r.length}")
    print(f"tcpSemER: {total.error_rate:.2%} [errors={total.errors:.2f}, "
          f"length={total.length}, sessions={len(res)}, collar={a.collar}]")

    if a.per_reco_out:
        with open(a.per_reco_out, "w") as f:
            json.dump({k: row(v) for k, v in sorted(res.items())}, f, indent=2)
    if a.average_out:
        with open(a.average_out, "w") as f:
            json.dump(row(total), f, indent=2)
    if a.segments_out:
        with open(a.segments_out, "w") as f:
            for sess, r in sorted(res.items()):
                for s in r.segments:
                    f.write(json.dumps({"session_id": sess, **dataclasses.asdict(s)}) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
