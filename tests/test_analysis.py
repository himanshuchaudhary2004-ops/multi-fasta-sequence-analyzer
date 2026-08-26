"""Tests for the computational helpers in app.py.

The Streamlit UI is intentionally not imported during tests. We load the pure
analysis functions from app.py's AST so CI can test the real implementation
without starting a Streamlit server.
"""
from pathlib import Path
import ast
import io
import math  # noqa: F401
import re  # noqa: F401
from collections import Counter  # noqa: F401

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner  # noqa: F401


APP = Path(__file__).resolve().parents[1] / "app.py"


def load_analysis_namespace():
    tree = ast.parse(APP.read_text(encoding="utf-8"))

    # Keep imports required by the helper functions, but exclude UI-only
    # dependencies and execution code.
    allowed_import_names = {
        "io", "math", "re", "Counter", "Path", "np", "pd",
        "SeqIO", "Seq", "PairwiseAligner",
    }

    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = {
                alias.asname or alias.name.split('.')[0]
                for alias in node.names
            }
            if names & allowed_import_names:
                body.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body.append(node)
        elif isinstance(node, ast.Assign):
            # Keep analysis constants such as DNA and restriction enzymes.
            targets = {
                t.id for t in node.targets if isinstance(t, ast.Name)
            }
            if targets & {"DNA", "RNA", "AMBIGUOUS", "PROTEIN",
                          "STOP_CODONS", "RESTRICTION_ENZYMES"}:
                body.append(node)

    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {
        "__file__": str(APP),
        "io": io,
        "math": math,
        "re": re,
        "Counter": Counter,
        "Path": Path,
        "pd": pd,
        "SeqIO": SeqIO,
        "Seq": Seq,
        "PairwiseAligner": PairwiseAligner,
    }
    exec(compile(module, str(APP), "exec"), namespace)
    return namespace


N = load_analysis_namespace()


def test_clean_sequence_and_detect_type():
    assert N["clean_sequence"](" a t g c\n") == "ATGC"
    assert N["detect_type"]("ATGCNN") == "DNA"
    assert N["detect_type"]("AUGC") == "RNA"
    assert N["detect_type"]("MKWVTF") == "Protein"


def test_composition_metrics():
    assert N["gc_percent"]("ATGC") == 50.0
    assert N["at_percent"]("ATGC") == 50.0
    assert N["gc_skew"]("GGCC") == 0.0
    assert N["at_skew"]("AATT") == 0.0


def test_sequence_complexity_helpers():
    assert N["shannon_entropy"]("AAAA") == 0.0
    assert N["longest_homopolymer"]("AATTTCC") == ("T", 3)
    assert N["ambiguous_count"]("ATGN") == 1


def test_quality_score_returns_valid_result():
    score, flags = N["quality_score"]("ATGC" * 30)
    assert 0 <= score <= 100
    assert isinstance(flags, list)


def test_kmer_and_motif_analysis():
    kt = N["kmer_table"]("ATGATG", 3)
    assert int(kt.loc[kt["k-mer"] == "ATG", "Count"].iloc[0]) == 2
    assert N["motif_positions"]("AATGATG", "ATG") == [2, 5]


def test_codon_usage():
    cu = N["codon_usage"]("ATGATGAAA")
    assert set(cu["Codon"]) == {"ATG", "AAA"}


def test_orf_detection():
    result = N["longest_orf"]("CCCATGAAATAGCCC", min_aa=2)
    assert result is not None
    assert result["AA length"] == 2
    assert result["Protein"] == "MK"


def test_six_frame_translation():
    result = N["six_frame_translation"]("ATGAAATAG")
    assert len(result) == 6
    assert set(result["Strand"]) == {"+", "-"}


def test_alignment_and_consensus():
    aligned = N["simple_global_align_many"](["ATGC", "ATGGC"])
    assert len(aligned) == 2
    consensus = N["consensus_from_alignment"](aligned)
    assert consensus
    conservation = N["conservation_table"](aligned)
    assert not conservation.empty


def test_variant_and_distance_analysis():
    variants = N["variant_table"]("ATGC", "ATGT")
    assert len(variants) == 1
    assert variants.iloc[0]["Change"] == "C>T"

    records = list(SeqIO.parse(
        str(Path(__file__).resolve().parents[1] / "sample_sequences.fasta"),
        "fasta",
    ))
    dist = N["pairwise_distance_matrix"](records)
    assert dist.shape == (len(records), len(records))
    assert (dist.values.diagonal() == 0).all()


def test_advanced_helpers():
    assert N["reverse_complement"]("ATGC") == "GCAT"
    assert N["dna_to_rna"]("ATGC") == "AUGC"
    assert N["rna_to_dna"]("AUGC") == "ATGC"
    assert N["cpg_count"]("ACGCGT") == 2


def test_restriction_table_and_report():
    table = N["restriction_table"]("GAATTC" * 2)
    eco = table.loc[table["Enzyme"] == "EcoRI"].iloc[0]
    assert int(eco["Sites"]) == 2

    records = list(SeqIO.parse(
        str(Path(__file__).resolve().parents[1] / "sample_sequences.fasta"),
        "fasta",
    ))
    df = N["build_dataframe"](records)
    report = N["make_report"](df)
    assert "MULTI-FASTA SEQUENCE ANALYSIS REPORT" in report
    assert f"Sequences: {len(records)}" in report


def test_sample_fasta_is_parseable():
    fasta = Path(__file__).resolve().parents[1] / "sample_sequences.fasta"
    records = list(SeqIO.parse(str(fasta), "fasta"))
    assert len(records) >= 2
    assert all(len(str(r.seq)) > 0 for r in records)
