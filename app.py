import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from Bio import SeqIO
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Multi-FASTA Sequence Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.25rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }
    .metric-note {
        font-size: 0.8rem;
        color: #777;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🧬 Multi-FASTA Sequence Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Python bioinformatics dashboard for sequence composition, GC analysis, statistics and visualization.</div>',
    unsafe_allow_html=True,
)


# ============================================================
# Helper functions
# ============================================================

DNA_BASES = set("ATGC")


def clean_sequence(seq: str) -> str:
    """Remove whitespace and convert a sequence to uppercase."""
    return "".join(str(seq).split()).upper()


def analyze_record(record, low_gc: float, high_gc: float) -> dict:
    """Calculate sequence-level statistics."""
    seq = clean_sequence(record.seq)
    length = len(seq)

    counts = {base: seq.count(base) for base in "ATGC"}
    valid = sum(counts.values())
    invalid = length - valid

    if valid:
        gc_pct = (counts["G"] + counts["C"]) / valid * 100
        at_pct = (counts["A"] + counts["T"]) / valid * 100
    else:
        gc_pct = np.nan
        at_pct = np.nan

    if valid and (counts["G"] + counts["C"]) > 0:
        gc_skew = (counts["G"] - counts["C"]) / (counts["G"] + counts["C"])
    else:
        gc_skew = np.nan

    if valid and (counts["A"] + counts["T"]) > 0:
        at_skew = (counts["A"] - counts["T"]) / (counts["A"] + counts["T"])
    else:
        at_skew = np.nan

    if np.isnan(gc_pct):
        classification = "No valid DNA bases"
    elif gc_pct < low_gc:
        classification = "AT-rich"
    elif gc_pct > high_gc:
        classification = "GC-rich"
    else:
        classification = "Balanced"

    return {
        "Sequence ID": record.id,
        "Description": record.description,
        "Length (bp)": length,
        "A": counts["A"],
        "T": counts["T"],
        "G": counts["G"],
        "C": counts["C"],
        "GC %": round(gc_pct, 2) if not np.isnan(gc_pct) else np.nan,
        "AT %": round(at_pct, 2) if not np.isnan(at_pct) else np.nan,
        "GC Skew": round(gc_skew, 4) if not np.isnan(gc_skew) else np.nan,
        "AT Skew": round(at_skew, 4) if not np.isnan(at_skew) else np.nan,
        "Valid Bases": valid,
        "Invalid Bases": invalid,
        "Classification": classification,
    }


def parse_fasta(uploaded_file, low_gc: float, high_gc: float):
    """Parse an uploaded FASTA file and return records, sequences and results."""
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8", errors="replace")

    records = list(SeqIO.parse(io.StringIO(text), "fasta"))

    results = []
    sequences = {}

    for record in records:
        seq = clean_sequence(record.seq)
        sequences[record.id] = seq
        results.append(analyze_record(record, low_gc, high_gc))

    return records, sequences, pd.DataFrame(results)


def sliding_gc(seq: str, window: int, step: int) -> pd.DataFrame:
    """Calculate GC% in a sliding window."""
    seq = clean_sequence(seq)
    rows = []

    if len(seq) < window:
        return pd.DataFrame(columns=["Start", "End", "GC %"])

    for start in range(0, len(seq) - window + 1, step):
        window_seq = seq[start:start + window]
        valid = sum(window_seq.count(base) for base in "ATGC")
        gc = window_seq.count("G") + window_seq.count("C")
        gc_pct = (gc / valid * 100) if valid else np.nan

        rows.append(
            {
                "Start": start + 1,
                "End": start + window,
                "GC %": round(gc_pct, 2) if not np.isnan(gc_pct) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def make_excel(df: pd.DataFrame, summary: pd.DataFrame) -> bytes:
    """Create an Excel workbook in memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sequence Results")
        summary.to_excel(writer, index=False, sheet_name="Summary")
    output.seek(0)
    return output.getvalue()


def make_fasta(records) -> bytes:
    """Return the parsed FASTA records as normalized FASTA bytes."""
    output = io.StringIO()
    SeqIO.write(records, output, "fasta")
    return output.getvalue().encode("utf-8")


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("⚙️ Analysis Settings")

    low_gc = st.number_input(
        "AT-rich threshold (%)",
        min_value=0.0,
        max_value=100.0,
        value=45.0,
        step=1.0,
        help="Sequences below this GC% are classified as AT-rich.",
    )

    high_gc = st.number_input(
        "GC-rich threshold (%)",
        min_value=0.0,
        max_value=100.0,
        value=55.0,
        step=1.0,
        help="Sequences above this GC% are classified as GC-rich.",
    )

    st.divider()

    st.markdown("### Sliding-window GC")
    window = st.number_input(
        "Window size (bp)",
        min_value=10,
        max_value=1000000,
        value=100,
        step=10,
    )
    step = st.number_input(
        "Step size (bp)",
        min_value=1,
        max_value=1000000,
        value=25,
        step=5,
    )

    st.divider()
    st.caption("GC% is calculated as (G + C) / (A + T + G + C) × 100. "
               "Invalid/ambiguous characters are excluded from the denominator.")


if low_gc >= high_gc:
    st.error("The AT-rich threshold must be lower than the GC-rich threshold.")
    st.stop()


# ============================================================
# File upload
# ============================================================

uploaded_file = st.file_uploader(
    "📁 Upload a Multi-FASTA file",
    type=["fasta", "fa", "fna", "fas"],
    help="Accepted formats: .fasta, .fa, .fna, .fas",
)

if uploaded_file is None:
    st.info("Upload a FASTA file to begin the analysis.")

    st.markdown("### Example FASTA format")
    st.code(
        """>Gene_1
ATGCGTACGTAGCTAGCTAGCTAGCGCGCGATATATCGCG
>Gene_2
ATATATATCGCGCGCGATATCGCGATATATCGCGCG
>Gene_3
GGCGCGCGCGATCGCGCGATCGATCGCGCGATCGCGCGCG""",
        language="text",
    )

    st.markdown("### What this app calculates")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Sequence metrics**")
        st.write("Length, A/T/G/C counts, valid and invalid bases")
    with c2:
        st.markdown("**Composition**")
        st.write("GC%, AT%, GC skew, AT skew and GC classification")
    with c3:
        st.markdown("**Visualization**")
        st.write("Bar charts, histograms, scatter plots, box plots and sliding-window GC")
    st.stop()


# ============================================================
# Parse data
# ============================================================

try:
    records, sequences, df = parse_fasta(uploaded_file, low_gc, high_gc)
except Exception as exc:
    st.error(f"Could not read the FASTA file: {exc}")
    st.stop()

if df.empty:
    st.error("No FASTA records were found in the uploaded file.")
    st.stop()

duplicate_ids = df.loc[df["Sequence ID"].duplicated(), "Sequence ID"].tolist()

# ============================================================
# Overview metrics
# ============================================================

valid_df = df[df["Valid Bases"] > 0].copy()

total_sequences = len(df)
total_bp = int(df["Length (bp)"].sum())
average_length = float(df["Length (bp)"].mean())
average_gc = float(valid_df["GC %"].mean()) if not valid_df.empty else np.nan
min_gc = float(valid_df["GC %"].min()) if not valid_df.empty else np.nan
max_gc = float(valid_df["GC %"].max()) if not valid_df.empty else np.nan

st.subheader("📌 Overview")

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Sequences", f"{total_sequences:,}")
m2.metric("Total bp", f"{total_bp:,}")
m3.metric("Average length", f"{average_length:,.1f} bp")
m4.metric("Average GC", f"{average_gc:.2f}%" if not np.isnan(average_gc) else "N/A")
m5.metric(
    "GC range",
    f"{min_gc:.1f}–{max_gc:.1f}%" if not np.isnan(min_gc) else "N/A",
)

if duplicate_ids:
    st.warning(
        f"Duplicate sequence IDs detected: {', '.join(duplicate_ids[:10])}"
        + (" ..." if len(duplicate_ids) > 10 else "")
    )

if int(df["Invalid Bases"].sum()) > 0:
    st.warning(
        f"Found {int(df['Invalid Bases'].sum()):,} characters outside A/T/G/C. "
        "These characters are excluded from GC% and AT% denominators."
    )



# ============================================================
# Advanced analysis helpers
# ============================================================

def sequence_type(seq):
    s = clean_sequence(seq)
    if not s:
        return "Empty"
    chars = set(s)
    if chars <= set("ATGCNRYKMSWBDHV"):
        return "DNA"
    if chars <= set("AUGCNRYKMSWBDHV"):
        return "RNA"
    if chars <= set("ACDEFGHIKLMNPQRSTVWYBXZJUO*"):
        return "Protein"
    return "Mixed/Unknown"


def shannon_entropy(seq):
    s = clean_sequence(seq)
    if not s:
        return 0.0
    counts = pd.Series(list(s)).value_counts()
    p = counts / len(s)
    return float(-(p * np.log2(p)).sum())


def longest_homopolymer(seq):
    s = clean_sequence(seq)
    if not s:
        return "", 0
    best_base, best_len = s[0], 1
    cur_base, cur_len = s[0], 1
    for b in s[1:]:
        if b == cur_base:
            cur_len += 1
        else:
            if cur_len > best_len:
                best_base, best_len = cur_base, cur_len
            cur_base, cur_len = b, 1
    if cur_len > best_len:
        best_base, best_len = cur_base, cur_len
    return best_base, best_len


def kmer_table(seq, k):
    s = clean_sequence(seq)
    if len(s) < k:
        return pd.DataFrame(columns=["k-mer", "Count", "Frequency (%)"])
    counts = pd.Series([s[i:i+k] for i in range(len(s)-k+1)]).value_counts()
    out = counts.rename_axis("k-mer").reset_index(name="Count")
    out["Frequency (%)"] = 100 * out["Count"] / out["Count"].sum()
    return out


def motif_hits(seq, motif):
    import re
    s = clean_sequence(seq)
    motif = motif.upper().strip()
    if not motif:
        return []
    try:
        return [m.start()+1 for m in re.finditer(f"(?={motif})", s)]
    except re.error:
        return []


def codon_usage(seq):
    s = clean_sequence(seq)
    codons = [s[i:i+3] for i in range(0, len(s)-2, 3)]
    codons = [c for c in codons if set(c) <= set("ATGC")]
    if not codons:
        return pd.DataFrame(columns=["Codon", "Count", "Frequency (%)"])
    counts = pd.Series(codons).value_counts()
    out = counts.rename_axis("Codon").reset_index(name="Count")
    out["Frequency (%)"] = 100 * out["Count"] / len(codons)
    return out


def restriction_sites(seq):
    import re
    enzymes = {
        "EcoRI": "GAATTC", "BamHI": "GGATCC", "HindIII": "AAGCTT",
        "PstI": "CTGCAG", "XhoI": "CTCGAG", "NotI": "GCGGCCGC",
        "SmaI": "CCCGGG"
    }
    s = clean_sequence(seq)
    rows = []
    for enzyme, motif in enzymes.items():
        positions = [m.start()+1 for m in re.finditer(f"(?={motif})", s)]
        rows.append({
            "Enzyme": enzyme,
            "Recognition site": motif,
            "Sites": len(positions),
            "Positions": ", ".join(map(str, positions[:100])) or "None"
        })
    return pd.DataFrame(rows)


def simple_qc_score(seq):
    s = clean_sequence(seq)
    if not s:
        return 0, ["Empty sequence"]
    score = 100
    flags = []
    invalid = sum(b not in "ATGC" for b in s)
    if invalid:
        score -= min(30, invalid)
        flags.append(f"{invalid} non-ACGT bases")
    if len(s) < 100:
        score -= 5
        flags.append("Short sequence")
    _, hp = longest_homopolymer(s)
    if hp >= 10:
        score -= 10
        flags.append(f"Homopolymer length {hp}")
    return max(0, score), flags or ["PASS"]


# ============================================================
# Tabs
# ============================================================

tab_overview, tab_table, tab_plots, tab_window, tab_qc, tab_tools, tab_report, tab_download = st.tabs(
    ["📊 Overview", "🧾 Sequence Table", "📈 Visualizations", "🔬 Sliding GC",
     "🧹 Research QC", "🧬 Sequence Tools", "📄 Report", "⬇️ Downloads"]
)


# ============================================================
# Overview tab
# ============================================================

with tab_overview:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### GC classification")

        class_counts = (
            df["Classification"]
            .value_counts()
            .rename_axis("Classification")
            .reset_index(name="Count")
        )

        fig_class = px.pie(
            class_counts,
            names="Classification",
            values="Count",
            hole=0.45,
            title="Sequence classification by GC content",
        )
        fig_class.update_layout(margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_class, use_container_width=True)

    with c2:
        st.markdown("#### Overall nucleotide composition")

        base_counts = pd.DataFrame(
            {
                "Base": ["A", "T", "G", "C"],
                "Count": [
                    int(df["A"].sum()),
                    int(df["T"].sum()),
                    int(df["G"].sum()),
                    int(df["C"].sum()),
                ],
            }
        )

        fig_base = px.bar(
            base_counts,
            x="Base",
            y="Count",
            text_auto=True,
            title="Total A/T/G/C counts",
        )
        fig_base.update_layout(margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_base, use_container_width=True)

    st.markdown("#### Summary statistics")

    summary = pd.DataFrame(
        {
            "Statistic": [
                "Total sequences",
                "Total base pairs",
                "Average sequence length",
                "Minimum sequence length",
                "Maximum sequence length",
                "Average GC%",
                "Minimum GC%",
                "Maximum GC%",
                "Average AT%",
                "Total invalid bases",
            ],
            "Value": [
                total_sequences,
                total_bp,
                f"{average_length:.2f} bp",
                f"{int(df['Length (bp)'].min())} bp",
                f"{int(df['Length (bp)'].max())} bp",
                f"{average_gc:.2f}%" if not np.isnan(average_gc) else "N/A",
                f"{min_gc:.2f}%" if not np.isnan(min_gc) else "N/A",
                f"{max_gc:.2f}%" if not np.isnan(max_gc) else "N/A",
                f"{valid_df['AT %'].mean():.2f}%" if not valid_df.empty else "N/A",
                int(df["Invalid Bases"].sum()),
            ],
        }
    )

    st.dataframe(summary, use_container_width=True, hide_index=True)


# ============================================================
# Table tab
# ============================================================

with tab_table:
    st.subheader("Sequence-level analysis")

    display_df = df.copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "GC %": st.column_config.NumberColumn(format="%.2f"),
            "AT %": st.column_config.NumberColumn(format="%.2f"),
            "GC Skew": st.column_config.NumberColumn(format="%.4f"),
            "AT Skew": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    st.caption(
        "GC skew = (G − C)/(G + C); AT skew = (A − T)/(A + T). "
        "These metrics are undefined when the corresponding denominator is zero."
    )


# ============================================================
# Visualization tab
# ============================================================

with tab_plots:
    st.subheader("Interactive visualizations")

    c1, c2 = st.columns(2)

    with c1:
        fig_gc = px.bar(
            df,
            x="Sequence ID",
            y="GC %",
            color="Classification",
            hover_data=["Length (bp)", "AT %"],
            title="GC content comparison",
        )
        fig_gc.update_yaxes(range=[0, 100], title="GC content (%)")
        fig_gc.update_layout(xaxis_title="Sequence ID")
        st.plotly_chart(fig_gc, use_container_width=True)

    with c2:
        fig_len = px.bar(
            df,
            x="Sequence ID",
            y="Length (bp)",
            color="Classification",
            hover_data=["GC %"],
            title="Sequence length comparison",
        )
        fig_len.update_layout(xaxis_title="Sequence ID", yaxis_title="Length (bp)")
        st.plotly_chart(fig_len, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig_hist = px.histogram(
            valid_df,
            x="GC %",
            nbins=20,
            title="GC content distribution",
        )
        fig_hist.update_xaxes(range=[0, 100], title="GC content (%)")
        fig_hist.update_yaxes(title="Number of sequences")
        st.plotly_chart(fig_hist, use_container_width=True)

    with c4:
        fig_box = px.box(
            valid_df,
            y="GC %",
            points="outliers",
            title="GC content spread and outliers",
        )
        fig_box.update_yaxes(range=[0, 100], title="GC content (%)")
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("#### GC content vs sequence length")

    if len(valid_df) >= 2 and valid_df["Length (bp)"].nunique() > 1:
        x = valid_df["Length (bp)"].to_numpy(dtype=float)
        y = valid_df["GC %"].to_numpy(dtype=float)

        regression = linregress(x, y)

        fig_scatter = px.scatter(
            valid_df,
            x="Length (bp)",
            y="GC %",
            color="Classification",
            hover_name="Sequence ID",
            title="Relationship between sequence length and GC content",
        )

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = regression.intercept + regression.slope * x_line

        fig_scatter.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Linear fit",
            )
        )
        fig_scatter.update_yaxes(range=[0, 100])
        st.plotly_chart(fig_scatter, use_container_width=True)

        r2 = regression.rvalue ** 2
        st.info(
            f"Pearson r = {regression.rvalue:.3f} | "
            f"R² = {r2:.3f} | p-value = {regression.pvalue:.3g}"
        )
    else:
        st.info("At least two sequences with different lengths are required for a correlation plot.")

    st.markdown("#### GC skew and AT skew")

    skew_df = df[["Sequence ID", "GC Skew", "AT Skew"]].melt(
        id_vars="Sequence ID",
        var_name="Metric",
        value_name="Value",
    )

    fig_skew = px.bar(
        skew_df,
        x="Sequence ID",
        y="Value",
        color="Metric",
        barmode="group",
        title="Sequence-level nucleotide skew",
    )
    st.plotly_chart(fig_skew, use_container_width=True)


# ============================================================
# Sliding-window tab
# ============================================================

with tab_window:
    st.subheader("Sliding-window GC analysis")

    sequence_id = st.selectbox(
        "Select a sequence",
        options=list(sequences.keys()),
    )

    selected_seq = sequences[sequence_id]

    st.write(
        f"**{sequence_id}** — {len(selected_seq):,} bp"
    )

    if len(selected_seq) < window:
        st.warning(
            f"The selected sequence is shorter than the {window}-bp window. "
            "Reduce the window size in the sidebar."
        )
    else:
        window_df = sliding_gc(selected_seq, int(window), int(step))

        fig_window = px.line(
            window_df,
            x="Start",
            y="GC %",
            markers=False,
            title=f"Sliding-window GC content: {sequence_id}",
        )
        fig_window.update_yaxes(range=[0, 100], title="GC content (%)")
        fig_window.update_xaxes(title="Window start position (bp)")
        st.plotly_chart(fig_window, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Minimum window GC", f"{window_df['GC %'].min():.2f}%")
        c2.metric("Maximum window GC", f"{window_df['GC %'].max():.2f}%")
        c3.metric("Mean window GC", f"{window_df['GC %'].mean():.2f}%")

        st.dataframe(window_df, use_container_width=True, hide_index=True)



# ============================================================
# Research QC tab
# ============================================================

with tab_qc:
    st.subheader("Research QC & Sequence Validation")
    qc_rows = []
    for sid, seq in sequences.items():
        score, flags = simple_qc_score(seq)
        qc_rows.append({
            "Sequence ID": sid,
            "Type": sequence_type(seq),
            "Length (bp)": len(seq),
            "GC %": round(gc_percent(seq), 2),
            "Entropy": round(shannon_entropy(seq), 3),
            "Quality Score": score,
            "QC Status": "PASS" if score >= 90 else ("REVIEW" if score >= 70 else "FAIL"),
            "Flags": "; ".join(flags),
        })
    qc_df = pd.DataFrame(qc_rows)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean QC score", f"{qc_df['Quality Score'].mean():.1f}/100")
    c2.metric("PASS", int((qc_df["QC Status"] == "PASS").sum()))
    c3.metric("REVIEW", int((qc_df["QC Status"] == "REVIEW").sum()))
    c4.metric("FAIL", int((qc_df["QC Status"] == "FAIL").sum()))

    st.dataframe(qc_df, use_container_width=True, hide_index=True)
    fig = px.bar(qc_df, x="Sequence ID", y="Quality Score", color="QC Status",
                 title="Per-sequence quality score")
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    comp = df.set_index("Sequence ID")[["A", "T", "G", "C"]]
    fig = px.imshow(comp.T, text_auto=True, aspect="auto", title="A/T/G/C Count Heatmap")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(df, x="Length (bp)", nbins=20,
                                     title="Sequence length distribution"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(px.histogram(valid_df, x="GC %", nbins=20,
                                     title="GC% distribution"),
                        use_container_width=True)


# ============================================================
# Sequence Tools tab
# ============================================================

with tab_tools:
    st.subheader("Sequence Tools")
    tool_id = st.selectbox("Select sequence", list(sequences.keys()), key="tool_id")
    seq = sequences[tool_id]

    st.markdown("#### Motif / pattern search")
    motif = st.text_input("Motif", value="ATG", key="motif")
    hits = motif_hits(seq, motif)
    st.write(f"**{len(hits)}** matches")
    if hits:
        st.write("1-based positions:", hits[:200])

    st.markdown("#### k-mer analysis")
    k = st.selectbox("k-mer size", [2, 3, 4, 5, 6], index=1)
    kt = kmer_table(seq, k)
    if not kt.empty:
        n = st.slider("Top k-mers", 5, min(50, len(kt)), min(20, len(kt)))
        st.plotly_chart(px.bar(kt.head(n), x="k-mer", y="Count",
                               title=f"Top {n} {k}-mers"),
                        use_container_width=True)
        st.dataframe(kt, use_container_width=True, hide_index=True)

    st.markdown("#### Codon usage")
    cu = codon_usage(seq)
    if not cu.empty:
        st.plotly_chart(px.bar(cu, x="Codon", y="Count", title="Codon usage"),
                        use_container_width=True)
        st.dataframe(cu, use_container_width=True, hide_index=True)
    else:
        st.info("Codon usage requires an unambiguous DNA sequence.")

    st.markdown("#### Restriction-site screening")
    st.dataframe(restriction_sites(seq), use_container_width=True, hide_index=True)

    st.markdown("#### Oligonucleotide Wallace-rule Tm")
    oligo = st.text_input("DNA oligo", value=seq[:20], key="oligo")
    clean_oligo = clean_sequence(oligo)
    if clean_oligo and set(clean_oligo) <= set("ATGC"):
        tm = 2 * (clean_oligo.count("A") + clean_oligo.count("T")) + 4 * (clean_oligo.count("G") + clean_oligo.count("C"))
        st.metric("Estimated Tm", f"{tm:.1f} °C")
        st.caption("Simple Wallace-rule estimate; not a full thermodynamic primer-design calculation.")


# ============================================================
# Automated report tab
# ============================================================

with tab_report:
    st.subheader("Automated Analysis Report")
    total_bases = int(df["Length (bp)"].sum())
    mean_len = df["Length (bp)"].mean()
    mean_gc = valid_df["GC %"].mean() if not valid_df.empty else np.nan

    report = f"""MULTI-FASTA SEQUENCE ANALYSIS REPORT
====================================

Dataset
-------
Total sequences: {len(df)}
Total bases: {total_bases:,}
Mean sequence length: {mean_len:,.2f} bp
Minimum length: {int(df['Length (bp)'].min())} bp
Maximum length: {int(df['Length (bp)'].max())} bp
Mean GC%: {mean_gc:.2f}%

Classification
--------------
{df['Classification'].value_counts().to_string()}

Quality Control
---------------
Mean QC score: {qc_df['Quality Score'].mean():.1f}/100
PASS: {(qc_df['QC Status'] == 'PASS').sum()}
REVIEW: {(qc_df['QC Status'] == 'REVIEW').sum()}
FAIL: {(qc_df['QC Status'] == 'FAIL').sum()}

Interpretation note
-------------------
QC indicators are screening metrics. Biological interpretation should consider
organism, genomic region, sequencing platform, assembly quality and study design.
"""
    st.text_area("Report preview", report, height=450)
    st.download_button("⬇️ Download report", report.encode("utf-8"),
                       file_name="multi_fasta_analysis_report.txt",
                       mime="text/plain", use_container_width=True)

    report_table = pd.DataFrame({
        "Metric": ["Sequences", "Total bases", "Mean length", "Mean GC%", "Mean QC score"],
        "Value": [len(df), total_bases, round(mean_len, 2), round(mean_gc, 2),
                  round(qc_df["Quality Score"].mean(), 2)]
    })
    st.dataframe(report_table, use_container_width=True, hide_index=True)


# ============================================================
# Downloads tab
# ============================================================

with tab_download:
    st.subheader("Download analysis results")

    summary = pd.DataFrame(
        {
            "Statistic": [
                "Total sequences",
                "Total base pairs",
                "Average sequence length",
                "Average GC%",
                "Minimum GC%",
                "Maximum GC%",
                "Total invalid bases",
            ],
            "Value": [
                total_sequences,
                total_bp,
                round(average_length, 2),
                round(average_gc, 2) if not np.isnan(average_gc) else None,
                round(min_gc, 2) if not np.isnan(min_gc) else None,
                round(max_gc, 2) if not np.isnan(max_gc) else None,
                int(df["Invalid Bases"].sum()),
            ],
        }
    )

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    excel_bytes = make_excel(df, summary)
    fasta_bytes = make_fasta(records)

    st.download_button(
        "⬇️ Download CSV",
        data=csv_bytes,
        file_name="multi_fasta_analysis.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        "📊 Download Excel",
        data=excel_bytes,
        file_name="multi_fasta_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.download_button(
        "🧬 Download normalized FASTA",
        data=fasta_bytes,
        file_name="normalized_sequences.fasta",
        mime="text/plain",
        use_container_width=True,
    )

    st.markdown("### Output contents")
    st.write(
        "The CSV/Excel output contains sequence ID, description, length, "
        "A/T/G/C counts, GC%, AT%, GC skew, AT skew, valid bases, invalid bases "
        "and GC classification."
    )


# ============================================================
# Footer
# ============================================================

st.divider()
st.caption(
    "Multi-FASTA Sequence Analyzer • Python • Biopython • Pandas • Plotly • SciPy"
)
