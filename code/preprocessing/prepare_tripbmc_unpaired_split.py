"""Create a deterministic crossed-unpaired assignment for TriPBMC cells."""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path


MODALITIES = ("RNA", "ATAC", "Protein")
STRATUM_FIELDS = (
    "source",
    "condition",
    "technical_stratum",
    "canonical_cell_type",
)


def stable_key(seed, *parts):
    payload = "|".join([str(seed)] + [str(part) for part in parts])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Allocate cells within strata while satisfying exact modality quotas.
def assign_stratified(records, targets, seed=42):
    records = [dict(record) for record in records]
    if set(targets) != set(MODALITIES):
        raise ValueError("targets must contain exactly RNA, ATAC, and Protein")
    targets = {modality: int(targets[modality]) for modality in MODALITIES}
    if any(value < 0 for value in targets.values()):
        raise ValueError("targets must be non-negative")
    if sum(targets.values()) != len(records):
        raise ValueError(
            "target total %d does not match record total %d"
            % (sum(targets.values()), len(records))
        )
    cell_ids = [record["cell_id"] for record in records]
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("cell_id values must be unique")

    groups = defaultdict(list)
    for record in records:
        stratum = tuple(record[field] for field in STRATUM_FIELDS)
        groups[stratum].append(record)

    base_total = sum(len(group) // len(MODALITIES) for group in groups.values())
    remaining = {
        modality: targets[modality] - base_total for modality in MODALITIES
    }
    remainders = sum(len(group) % len(MODALITIES) for group in groups.values())
    if any(value < 0 for value in remaining.values()) or sum(remaining.values()) != remainders:
        raise ValueError("targets are incompatible with per-stratum balance")

    assigned = []
    for stratum in sorted(groups):
        group = sorted(
            groups[stratum], key=lambda row: stable_key(seed, row["cell_id"])
        )
        base, remainder = divmod(len(group), len(MODALITIES))
        counts = {modality: base for modality in MODALITIES}
        eligible = [modality for modality in MODALITIES if remaining[modality] > 0]
        if len(eligible) < remainder:
            raise ValueError("not enough positive modality quota for stratum remainder")
        candidates = sorted(
            eligible,
            key=lambda modality: (
                -remaining[modality],
                stable_key(seed, *stratum, modality),
            ),
        )
        for modality in candidates[:remainder]:
            counts[modality] += 1
            remaining[modality] -= 1

        offset = 0
        for modality in MODALITIES:
            stop = offset + counts[modality]
            for record in group[offset:stop]:
                record["modality"] = modality
                assigned.append(record)
            offset = stop
        if offset != len(group):
            raise AssertionError("stratum allocation did not consume every cell")

    if any(remaining.values()):
        raise ValueError("source modality quotas were not exhausted: %r" % remaining)
    observed = Counter(record["modality"] for record in assigned)
    if observed != Counter(targets):
        raise AssertionError("observed modality counts do not match targets")
    return sorted(
        assigned,
        key=lambda row: (row["source"], row.get("canonical_barcode", row["cell_id"])),
    )


def association_metrics(records):
    rows = list(records)
    if not rows:
        raise ValueError("association metrics require at least one record")
    sources = sorted({row["source"] for row in rows})
    modalities = [modality for modality in MODALITIES if any(row["modality"] == modality for row in rows)]
    table = {
        (source, modality): 0 for source in sources for modality in modalities
    }
    for row in rows:
        table[(row["source"], row["modality"])] += 1
    total = float(len(rows))
    source_totals = {
        source: sum(table[(source, modality)] for modality in modalities)
        for source in sources
    }
    modality_totals = {
        modality: sum(table[(source, modality)] for source in sources)
        for modality in modalities
    }
    mutual_information = 0.0
    chi_square = 0.0
    for source in sources:
        for modality in modalities:
            observed = float(table[(source, modality)])
            expected = source_totals[source] * modality_totals[modality] / total
            if observed > 0.0:
                p_joint = observed / total
                p_source = source_totals[source] / total
                p_modality = modality_totals[modality] / total
                mutual_information += p_joint * math.log(
                    p_joint / (p_source * p_modality)
                )
            if expected > 0.0:
                chi_square += (observed - expected) ** 2 / expected
    dimension = min(len(sources) - 1, len(modalities) - 1)
    cramers_v = math.sqrt(chi_square / (total * dimension)) if dimension > 0 else 0.0
    return {
        "mutual_information_nats": float(mutual_information),
        "chi_square": float(chi_square),
        "cramers_v": float(cramers_v),
    }


EXPECTED_CELLS = {
    "DOGMA_Control": 7468,
    "DOGMA_Stim": 5915,
    "TEA": 25517,
}
SOURCE_TARGETS = {
    "DOGMA_Control": {"RNA": 2490, "ATAC": 2489, "Protein": 2489},
    "DOGMA_Stim": {"RNA": 1971, "ATAC": 1972, "Protein": 1972},
    "TEA": {"RNA": 8506, "ATAC": 8506, "Protein": 8505},
}
CONDITIONS = {
    "DOGMA_Control": "Control",
    "DOGMA_Stim": "Stim",
    "TEA": "All",
}
OUTPUT_COLUMNS = (
    "cell_id",
    "source",
    "condition",
    "canonical_barcode",
    "technical_stratum",
    "source_cell_type",
    "canonical_cell_type",
    "modality",
)


def normalize_whitespace(value):
    return " ".join(str(value).split())


def canonicalize_adt_barcode(value):
    value = str(value).strip()
    if "." in value:
        base, suffix = value.rsplit(".", 1)
        if base and suffix:
            return base + "-" + suffix
    return value


def _read_lines(path):
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        values = [line.rstrip("\r\n") for line in handle if line.rstrip("\r\n")]
    if len(set(values)) != len(values):
        raise ValueError("duplicate values in %s" % path)
    return values


def _read_adt_barcodes(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), None)
    if not header or len(header) < 2:
        raise ValueError("ADT header is missing cell barcodes: %s" % path)
    values = [canonicalize_adt_barcode(value) for value in header[1:]]
    if len(set(values)) != len(values):
        raise ValueError("duplicate ADT barcodes in %s" % path)
    return values


def _read_labels(path):
    labels = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"barcode", "celltype"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("label file must contain barcode and celltype: %s" % path)
        for row in reader:
            barcode = str(row["barcode"]).strip()
            if barcode in labels:
                raise ValueError("duplicate label barcode %s in %s" % (barcode, path))
            label = normalize_whitespace(row["celltype"])
            if not barcode or not label:
                raise ValueError("empty barcode or celltype in %s" % path)
            labels[barcode] = label
    return labels


# Validate barcode and label agreement across all source modalities.
def load_records(source_root, expected_cells=None):
    root = Path(source_root)
    expected = dict(EXPECTED_CELLS if expected_cells is None else expected_cells)
    records = []
    for source in sorted(expected):
        data = root / "data" / source
        rna_barcodes = _read_lines(data / "RNA" / "barcodes.tsv")
        atac_barcodes = _read_lines(data / "ATAC" / "barcodes.tsv")
        adt_barcodes = _read_adt_barcodes(data / "ADT.csv")
        if rna_barcodes != atac_barcodes:
            raise ValueError("%s RNA/ATAC barcode mismatch" % source)
        if rna_barcodes != adt_barcodes:
            raise ValueError("%s RNA/ADT barcode mismatch after canonicalization" % source)
        if len(rna_barcodes) != int(expected[source]):
            raise ValueError(
                "%s expected %d cells but found %d"
                % (source, int(expected[source]), len(rna_barcodes))
            )
        labels = _read_labels(root / "labels" / (source + "_celltype.csv"))
        barcode_set = set(rna_barcodes)
        if set(labels) != barcode_set:
            raise ValueError(
                "%s label barcode mismatch: missing=%d extra=%d"
                % (source, len(barcode_set - set(labels)), len(set(labels) - barcode_set))
            )
        condition = CONDITIONS.get(source, "All")
        for barcode in rna_barcodes:
            if source == "TEA":
                suffix = barcode.rsplit("-", 1)[-1] if "-" in barcode else "unknown"
                technical_stratum = "barcode_suffix_" + suffix
            else:
                technical_stratum = "not_available"
            source_cell_type = labels[barcode]
            records.append(
                {
                    "cell_id": source + "::" + barcode,
                    "source": source,
                    "condition": condition,
                    "canonical_barcode": barcode,
                    "technical_stratum": technical_stratum,
                    "source_cell_type": source_cell_type,
                    "canonical_cell_type": normalize_whitespace(source_cell_type),
                }
            )
    cell_ids = [record["cell_id"] for record in records]
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("source-qualified cell_id values are not unique")
    return sorted(records, key=lambda row: (row["source"], row["canonical_barcode"]))


def input_fingerprint(source_root):
    root = Path(source_root)
    selected = []
    contract = root / "DATA_CONTRACT.json"
    if contract.exists():
        selected.append(contract)
    for source in sorted(EXPECTED_CELLS):
        selected.extend(
            [
                root / "data" / source / "RNA" / "barcodes.tsv",
                root / "data" / source / "ATAC" / "barcodes.tsv",
                root / "labels" / (source + "_celltype.csv"),
            ]
        )
    digest = hashlib.sha256()
    for path in selected:
        relative = str(path.relative_to(root)).replace(os.sep, "/")
        digest.update(relative.encode("utf-8") + b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    for source in sorted(EXPECTED_CELLS):
        adt = root / "data" / source / "ADT.csv"
        with adt.open("rb") as handle:
            header = handle.readline()
        digest.update(("data/%s/ADT.csv:header" % source).encode("utf-8") + b"\0")
        digest.update(header)
    return digest.hexdigest()


# Verify disjointness, balance, coverage, and source independence.
def build_summary(assigned, source_targets, seed, input_fingerprint):
    rows = [dict(row) for row in assigned]
    expected_total = sum(
        int(count)
        for targets in source_targets.values()
        for count in targets.values()
    )
    if len(rows) != expected_total:
        raise ValueError(
            "assignment row count %d does not match target total %d"
            % (len(rows), expected_total)
        )
    cell_ids = [row["cell_id"] for row in rows]
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("each biological cell must appear exactly once")

    source_by_modality = {}
    for source in sorted(source_targets):
        counts = Counter(
            row["modality"] for row in rows if row["source"] == source
        )
        observed = {modality: int(counts[modality]) for modality in MODALITIES}
        target = {modality: int(source_targets[source][modality]) for modality in MODALITIES}
        if observed != target:
            raise ValueError(
                "%s source modality counts %r do not match targets %r"
                % (source, observed, target)
            )
        source_by_modality[source] = observed

    modality_sets = {
        modality: {row["cell_id"] for row in rows if row["modality"] == modality}
        for modality in MODALITIES
    }
    pairwise_overlaps = {}
    for left_index, left in enumerate(MODALITIES):
        for right in MODALITIES[left_index + 1 :]:
            overlap = len(modality_sets[left] & modality_sets[right])
            pairwise_overlaps[left + "__" + right] = overlap
            if overlap:
                raise ValueError("modality cell sets overlap: %s/%s" % (left, right))

    stratum_counts = defaultdict(Counter)
    for row in rows:
        stratum = tuple(row[field] for field in STRATUM_FIELDS)
        stratum_counts[stratum][row["modality"]] += 1
    max_imbalance = 0
    for stratum, counts in stratum_counts.items():
        values = [counts[modality] for modality in MODALITIES]
        imbalance = max(values) - min(values)
        max_imbalance = max(max_imbalance, imbalance)
        if imbalance > 1:
            raise ValueError("stratum modality imbalance exceeds one: %r" % (stratum,))

    celltype_groups = defaultdict(list)
    for row in rows:
        celltype_groups[(row["source"], row["canonical_cell_type"])].append(row)
    coverage_failures = []
    for key, group in sorted(celltype_groups.items()):
        present = {row["modality"] for row in group}
        if len(group) >= len(MODALITIES) and present != set(MODALITIES):
            coverage_failures.append(
                {"source": key[0], "canonical_cell_type": key[1], "cells": len(group), "modalities": sorted(present)}
            )
    if coverage_failures:
        raise ValueError("source cell-type modality coverage failed: %r" % coverage_failures)

    metrics = association_metrics(rows)
    if metrics["mutual_information_nats"] >= 1e-4:
        raise ValueError("source/modality mutual information exceeds threshold")
    if metrics["cramers_v"] >= 1e-3:
        raise ValueError("source/modality Cramer's V exceeds threshold")

    modality_totals = Counter(row["modality"] for row in rows)
    return {
        "algorithm": "tripbmc_stratified_sha256_v1",
        "seed": int(seed),
        "paired_test_holdout": False,
        "strict_unpaired": True,
        "eligible_cells": len(rows),
        "input_fingerprint_sha256": str(input_fingerprint),
        "modality_totals": {
            modality: int(modality_totals[modality]) for modality in MODALITIES
        },
        "source_by_modality": source_by_modality,
        "source_modality_association": metrics,
        "pairwise_cell_id_overlaps": pairwise_overlaps,
        "strata": {
            "fields": list(STRATUM_FIELDS),
            "count": len(stratum_counts),
            "maximum_modality_count_difference": int(max_imbalance),
        },
        "source_celltype_groups_checked": len(celltype_groups),
        "source_celltype_coverage_failures": coverage_failures,
    }


def _write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Write the assignment atomically and record output hashes.
def publish_outputs(assigned, summary, output_dir):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite existing output: %s" % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp-" + uuid.uuid4().hex)
    if temporary.exists():
        raise FileExistsError("temporary output already exists: %s" % temporary)
    rows = sorted(
        [dict(row) for row in assigned],
        key=lambda row: (row["source"], row["canonical_barcode"]),
    )
    try:
        temporary.mkdir()
        manifest = temporary / "split_manifest.csv"
        _write_csv(manifest, rows)
        output_hashes = {"split_manifest.csv": _sha256_file(manifest)}
        for modality in MODALITIES:
            path = temporary / (modality + "_only.csv")
            _write_csv(path, [row for row in rows if row["modality"] == modality])
            output_hashes[path.name] = _sha256_file(path)
        written_summary = dict(summary)
        written_summary["manifest_sha256"] = output_hashes["split_manifest.csv"]
        written_summary["output_sha256"] = output_hashes
        (temporary / "split_summary.json").write_text(
            json.dumps(written_summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        readme = """# TriPBMC Crossed-Unpaired assignment\n\nThis directory contains a deterministic cell-to-modality assignment for all eligible cells in the source dataset.\n\n- No paired test holdout exists.\n- Every biological cell appears exactly once.\n- Labels are retained for stratification and evaluation only; they are not training covariates.\n- Source matrices remain in the immutable source package and are not duplicated here.\n"""
        (temporary / "README.md").write_text(readme, encoding="utf-8")
        os.replace(str(temporary), str(output))
    except Exception:
        if temporary.exists():
            shutil.rmtree(str(temporary))
        raise
    return written_summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    records = load_records(source_root, expected_cells=EXPECTED_CELLS)
    assigned = []
    for source in sorted(EXPECTED_CELLS):
        source_records = [row for row in records if row["source"] == source]
        assigned.extend(
            assign_stratified(source_records, SOURCE_TARGETS[source], seed=args.seed)
        )
    assigned = sorted(
        assigned, key=lambda row: (row["source"], row["canonical_barcode"])
    )
    summary = build_summary(
        assigned,
        source_targets=SOURCE_TARGETS,
        seed=args.seed,
        input_fingerprint=input_fingerprint(source_root),
    )
    written = publish_outputs(assigned, summary, Path(args.output_dir))
    print("published %d cells to %s" % (written["eligible_cells"], args.output_dir))
    print("modality totals: %s" % written["modality_totals"])
    print("source/modality association: %s" % written["source_modality_association"])
    print("manifest sha256: %s" % written["manifest_sha256"])


if __name__ == "__main__":
    main()
