"""Which sources can be proven at all, and how to verify the ones that cannot.

Grading a release-parity refactor needs three independent questions, not one:

1. Does the published row carry a join key?      -> gsmp.schema.join_key_quality
2. Is that key unique?                            -> gsmp.schema.join_key_quality
3. Is the ORIGINAL PRODUCER deterministic?        -> this module

Question 3 was missed in the first grading pass and is the reason three
sources "failed" a regression they could never have passed. Their legacy
producers seed with `random.Random(hash(<str>) & 0xFFFFFFFF)`:

    legacy/ingest_sparsh.py:78        hash(indenter)
    legacy/ingest_tacquad_full.py:88  hash(domain)
    legacy/extract_rtm_video.py:123   hash(fname)

CPython randomises `hash(str)` per process (SipHash keyed from PYTHONHASHSEED,
unset in this environment). Measured across three processes, `hash('flat')`
returned 3873258486, 1233441081 and 1360980178. The seed that produced the
release was never logged, so the release cannot be reproduced -- not by a
port, and not by re-running the original script either.

The rng drives both the baseline sample and the background-keep draw, so a
different seed shifts borderline filter outcomes as well as the background
picks. Every other producer uses a literal integer seed and is reproducible.

"Cannot prove exact" is not "cannot verify". A faithful port should differ
from the release by roughly as much as it differs from ITSELF under another
seed. That is what `evaluate_envelope` measures.
"""
from __future__ import annotations

import dataclasses
import itertools
from typing import Dict, List, Sequence, Set, Tuple

Key = Tuple[str, int]

#: source -> (file:line, what the hash is taken of)
HASH_SEEDED_PRODUCERS: Dict[str, Tuple[str, str]] = {
    "sparsh": ("legacy/ingest_sparsh.py:78", "hash(indenter)"),
    "tacquad": ("legacy/ingest_tacquad_full.py:88", "hash(domain)"),
    "real_tactile_mnist": ("legacy/extract_rtm_video.py:123", "hash(fname)"),
}


def producer_is_reproducible(source: str) -> bool:
    """False if the legacy producer's RNG seed cannot be recovered."""
    return source not in HASH_SEEDED_PRODUCERS


def why_not_reproducible(source: str) -> str:
    """Human-readable reason, with the citation. Empty for reproducible ones."""
    if producer_is_reproducible(source):
        return ""
    where, what = HASH_SEEDED_PRODUCERS[source]
    return (
        f"{where} seeds `random.Random({what} & 0xFFFFFFFF)`; CPython "
        f"randomises str hash per process, and the seed used for the release "
        f"was never recorded, so exact parity is impossible in principle"
    )


@dataclasses.dataclass(frozen=True)
class SeedEnvelope:
    """How our port's spread under reseeding compares to its gap vs the release."""

    source: str
    n_runs: int
    self_symdiffs: List[int]
    published_symdiff: int
    n_published: int
    mean_produced: float
    consistent: bool
    verdict: str
    capture_coverage: float


def evaluate_envelope(
    source: str,
    run_key_sets: Sequence[Set[Key]],
    published: Set[Key],
    restrict_to_covered_captures: bool = False,
) -> SeedEnvelope:
    """Compare our run-to-run spread against our gap to the published set.

    `run_key_sets` are kept-key sets from independent runs of the SAME port,
    each under a different process hash seed. If the published set is no
    further from our runs than they are from each other, the difference is
    attributable to the unrecoverable seed rather than to a porting error.

    `restrict_to_covered_captures` scopes `published` to the captures the runs
    actually visited. This is REQUIRED when the runs were produced with a
    `--limit`, because otherwise every unvisited capture counts as a missing
    key and the verdict is meaningless. It is also dangerous: scoping hides a
    port that drops whole captures, so `capture_coverage` is reported
    alongside and must be read with the verdict.
    """
    runs = [set(r) for r in run_key_sets]
    if len(runs) < 2:
        raise ValueError("need at least 2 runs to measure a seed envelope")

    published_captures = {c for c, _ in published}
    covered = {c for r in runs for c, _ in r}
    coverage = (
        len(covered & published_captures) / len(published_captures)
        if published_captures else 1.0
    )
    if restrict_to_covered_captures:
        published = {(c, i) for c, i in published if c in covered}

    self_symdiffs = [len(a ^ b) for a, b in itertools.combinations(runs, 2)]
    published_symdiff = min(len(r ^ published) for r in runs)
    mean_produced = sum(len(r) for r in runs) / len(runs)
    ceiling = max(self_symdiffs)

    floor = min(self_symdiffs)
    consistent = published_symdiff <= ceiling

    # "<= max" alone is a weak bar: with a wide self-spread almost anything
    # clears it. Report WHERE the gap falls in that spread, so a marginal
    # result cannot be read as a strong one.
    if published_symdiff < floor:
        verdict = (
            f"strong -- gap to release {published_symdiff} is below our own "
            f"minimum run-to-run spread {floor}: the port is closer to the "
            f"release than two of its own runs are to each other "
            f"(spread {floor}-{ceiling})"
        )
    elif consistent:
        verdict = (
            f"consistent with seed uncertainty -- gap to release "
            f"{published_symdiff} lies inside our own run-to-run spread "
            f"{floor}-{ceiling}; note a wide spread makes this a weak bar"
        )
    else:
        verdict = (
            f"gap to release {published_symdiff} exceeds our own run-to-run "
            f"spread {floor}-{ceiling} -- a real difference beyond the seed"
        )

    return SeedEnvelope(
        source=source,
        n_runs=len(runs),
        self_symdiffs=self_symdiffs,
        published_symdiff=published_symdiff,
        n_published=len(published),
        mean_produced=mean_produced,
        consistent=consistent,
        verdict=verdict,
        capture_coverage=coverage,
    )
