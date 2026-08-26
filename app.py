
import io
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from Bio import SeqIO
from Bio.Seq import Seq

st.set_page_config(page_title="Multi-FASTA Sequence Analyzer", page_icon="🧬", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.1rem;padding-bottom:2rem}
.hero{padding:1.4rem 1.6rem;border-radius:18px;border:1px solid rgba(120,120,120,.2);
background:linear-gradient(135deg,rgba(45,95,170,.12),rgba(55,170,135,.10));margin-bottom:1rem}
.hero h1{margin:0;font-size:2.1rem}.hero p{margin:.45rem 0 0;opacity:.76}
[data-testid="stMetric"]{border:1px solid rgba(120,120,120,.16);padding:.6rem;border-radius:12px}
.muted{opacity:.68;font-size:.86rem}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1>🧬 Multi-FASTA Sequence Analyzer</h1>
<p>Professional FASTA QC, sequence statistics, visualization, ORF screening, motif,
k-mer, codon, restriction-site, primer and comparative analysis.</p></div>""", unsafe_allow_html=True)

DNA=set("ATGC")
RNA=set("AUGC")
AMBIGUOUS=set("NRYKMSWBDHV")
PROTEIN=set("ACDEFGHIKLMNPQRSTVWYBXZJUO*")
STOP_CODONS={"TAA","TAG","TGA"}
RESTRICTION_ENZYMES={"EcoRI":"GAATTC","BamHI":"GGATCC","HindIII":"AAGCTT",
                     "PstI":"CTGCAG","XhoI":"CTCGAG","NotI":"GCGGCCGC","SmaI":"CCCGGG"}

def clean_sequence(seq):
    return re.sub(r"\s+","",str(seq).upper())

def detect_type(seq):
    s=clean_sequence(seq)
    if not s:return "Empty"
    chars=set(s)
    if chars <= DNA|AMBIGUOUS:return "DNA"
    if chars <= RNA|AMBIGUOUS:return "RNA"
    if chars <= PROTEIN:return "Protein"
    return "Mixed/Unknown"

def gc_percent(seq):
    s=clean_sequence(seq)
    return 100.0*(s.count("G")+s.count("C"))/len(s) if s else 0.0

def at_percent(seq):
    s=clean_sequence(seq)
    return 100.0*(s.count("A")+s.count("T"))/len(s) if s else 0.0

def gc_skew(seq):
    s=clean_sequence(seq); g=s.count("G"); c=s.count("C")
    return (g-c)/(g+c) if g+c else 0.0

def at_skew(seq):
    s=clean_sequence(seq); a=s.count("A"); t=s.count("T")
    return (a-t)/(a+t) if a+t else 0.0

def shannon_entropy(seq):
    s=clean_sequence(seq)
    if not s:return 0.0
    counts=Counter(s); n=len(s)
    return float(-sum((v/n)*math.log2(v/n) for v in counts.values()))

def longest_homopolymer(seq):
    s=clean_sequence(seq)
    if not s:return "",0
    best_base=cur_base=s[0]; best_len=cur_len=1
    for b in s[1:]:
        if b==cur_base:cur_len+=1
        else:
            if cur_len>best_len:best_base,best_len=cur_base,cur_len
            cur_base,cur_len=b,1
    if cur_len>best_len:best_base,best_len=cur_base,cur_len
    return best_base,best_len

def ambiguous_count(seq):
    return sum(b not in DNA for b in clean_sequence(seq))

def quality_score(seq):
    s=clean_sequence(seq)
    if not s:return 0,["Empty sequence"]
    score=100; flags=[]
    invalid=sum(b not in DNA for b in s)
    if invalid:score-=min(30,invalid);flags.append(f"{invalid} non-ACGT")
    if len(s)<100:score-=5;flags.append("short")
    _,hp=longest_homopolymer(s)
    if hp>=10:score-=10;flags.append(f"homopolymer {hp}")
    if len(set(s))<=2:score-=10;flags.append("low complexity")
    return max(0,score),flags or ["PASS"]

def gc_class(v):
    if v<40:return "Low (<40%)"
    if v<=60:return "Moderate (40–60%)"
    return "High (>60%)"

def sliding_metric(seq,window,metric):
    s=clean_sequence(seq)
    if len(s)<window:return pd.DataFrame(columns=["Position",metric])
    rows=[]
    for i in range(len(s)-window+1):
        w=s[i:i+window]
        val={"GC %":gc_percent,"GC Skew":gc_skew,"AT Skew":at_skew,
             "Entropy":shannon_entropy}[metric](w)
        rows.append({"Position":i+1,metric:val})
    return pd.DataFrame(rows)

def kmer_table(seq,k):
    s=clean_sequence(seq)
    if len(s)<k:return pd.DataFrame(columns=["k-mer","Count","Frequency (%)"])
    counts=Counter(s[i:i+k] for i in range(len(s)-k+1)); total=sum(counts.values())
    return pd.DataFrame([{"k-mer":x,"Count":n,"Frequency (%)":100*n/total} for x,n in counts.most_common()])

def motif_positions(seq,pattern):
    try:return [m.start()+1 for m in re.finditer(f"(?={pattern.strip().upper()})",clean_sequence(seq))]
    except re.error:return []

def codon_usage(seq):
    s=clean_sequence(seq)
    if not s or set(s)-DNA:return pd.DataFrame(columns=["Codon","Count","Frequency (%)"])
    codons=[s[i:i+3] for i in range(0,len(s)-2,3)]
    counts=Counter(codons); total=sum(counts.values())
    return pd.DataFrame([{"Codon":x,"Count":n,"Frequency (%)":100*n/total} for x,n in counts.most_common()])

def restriction_table(seq):
    s=clean_sequence(seq); rows=[]
    for enzyme,site in RESTRICTION_ENZYMES.items():
        pos=[m.start()+1 for m in re.finditer(f"(?={site})",s)]
        rows.append({"Enzyme":enzyme,"Recognition site":site,"Sites":len(pos),
                     "Positions":", ".join(map(str,pos[:100])) or "None"})
    return pd.DataFrame(rows)

def longest_orf(seq,min_aa=0):
    s=clean_sequence(seq)
    if set(s)-DNA:return None
    best=None
    for frame in range(3):
        start=None
        for i in range(frame,len(s)-2,3):
            codon=s[i:i+3]
            if start is None and codon=="ATG":start=i
            elif start is not None and codon in STOP_CODONS:
                aa=(i-start)//3
                if aa>=min_aa and (best is None or aa>best["AA length"]):
                    frag=s[start:i+3]
                    best={"Frame":frame+1,"Start":start+1,"Stop":i+3,
                          "NT length":len(frag),"AA length":aa,
                          "Protein":str(Seq(frag).translate(to_stop=True))}
                start=None
    return best

def six_frame_translation(seq):
    s=clean_sequence(seq)
    if set(s)-DNA:return pd.DataFrame()
    rows=[]
    for strand,dna in [("+",s),("-",str(Seq(s).reverse_complement()))]:
        for frame in range(3):
            p=str(Seq(dna[frame:]).translate(to_stop=False))
            rows.append({"Strand":strand,"Frame":f"{strand}{frame+1}","AA length":len(p),"Translation":p})
    return pd.DataFrame(rows)

def position_composition(records):
    max_len=max((len(clean_sequence(r.seq)) for r in records),default=0); rows=[]
    for pos in range(max_len):
        b=[clean_sequence(r.seq)[pos] for r in records if pos<len(clean_sequence(r.seq))]
        if b:
            n=len(b); rows.append({"Position":pos+1,"A %":100*b.count("A")/n,
                                   "T %":100*b.count("T")/n,"G %":100*b.count("G")/n,
                                   "C %":100*b.count("C")/n})
    return pd.DataFrame(rows)

def build_dataframe(records):
    rows=[]
    for r in records:
        s=clean_sequence(r.seq); a,t,g,c=[s.count(x) for x in "ATGC"]; score,flags=quality_score(s)
        hpbase,hplen=longest_homopolymer(s)
        rows.append({"ID":r.id,"Description":r.description,"Type":detect_type(s),"Length":len(s),
                     "A":a,"T":t,"G":g,"C":c,"GC %":round(gc_percent(s),3),"AT %":round(at_percent(s),3),
                     "GC Skew":round(gc_skew(s),5),"AT Skew":round(at_skew(s),5),
                     "Entropy":round(shannon_entropy(s),5),"Unique symbols":len(set(s)),
                     "Ambiguous bases":ambiguous_count(s),"GC Class":gc_class(gc_percent(s)),
                     "Longest homopolymer":f"{hpbase}{hplen}","QC Score":score,
                     "QC Status":"PASS" if score>=90 else ("REVIEW" if score>=70 else "FAIL"),
                     "QC Flags":"; ".join(flags)})
    return pd.DataFrame(rows)

def normalized_fasta(records):
    out=io.StringIO()
    for r in records:
        s=clean_sequence(r.seq);out.write(f">{r.description}\n")
        for i in range(0,len(s),80):out.write(s[i:i+80]+"\n")
    return out.getvalue().encode()

def make_report(df):
    return f"""MULTI-FASTA SEQUENCE ANALYSIS REPORT
====================================

Sequences: {len(df)}
Total bases/nt: {int(df["Length"].sum()):,}
Mean length: {df["Length"].mean():,.2f}
Minimum length: {int(df["Length"].min()) if len(df) else 0}
Maximum length: {int(df["Length"].max()) if len(df) else 0}
Mean GC%: {df["GC %"].mean():.2f}%

Sequence types
--------------
{df["Type"].value_counts().to_string()}

Quality control
---------------
Mean QC score: {df["QC Score"].mean():.2f}/100
PASS: {(df["QC Status"]=="PASS").sum()}
REVIEW: {(df["QC Status"]=="REVIEW").sum()}
FAIL: {(df["QC Status"]=="FAIL").sum()}

Interpretation note:
QC scores are computational screening indicators, not laboratory acceptance
criteria. Interpret results in the biological and sequencing context.
"""

# Input
if "records" not in st.session_state: st.session_state.records=[]
with st.sidebar:
    st.header("📂 FASTA Manager")
    uploaded=st.file_uploader("Upload FASTA file(s)",type=["fasta","fa","fas","fna","ffn","faa"],accept_multiple_files=True)
    if st.button("Clear dataset",use_container_width=True):
        st.session_state.records=[];st.rerun()
    st.divider()
    st.caption("Research Analysis Suite")
    st.caption("QC • composition • ORF • motifs • k-mers • reports")

    if uploaded:
        parsed=[];errors=[]
        for f in uploaded:
            try:
                raw=f.getvalue().decode("utf-8",errors="replace")
                rs=list(SeqIO.parse(io.StringIO(raw),"fasta"))
                for r in rs:
                    r.id=f"{f.name}:{r.id}";r.description=f"{f.name} | {r.description}"
                parsed.extend(rs)
            except Exception as exc: errors.append(f"{f.name}: {exc}")
        if errors: st.error("\n".join(errors))
        elif parsed: st.session_state.records=parsed

records=st.session_state.records
if not records:
    sample=Path(__file__).with_name("sample_sequences.fasta")
    if sample.exists():
        try: records=list(SeqIO.parse(str(sample),"fasta"));st.session_state.records=records
        except Exception: records=[]
if not records:
    st.info("Upload a FASTA file to begin.");st.stop()

df=build_dataframe(records)

m1,m2,m3,m4,m5,m6=st.columns(6)
m1.metric("Sequences",f"{len(df):,}");m2.metric("Total bases",f"{int(df['Length'].sum()):,}")
m3.metric("Mean length",f"{df['Length'].mean():,.0f}");m4.metric("Mean GC%",f"{df['GC %'].mean():.2f}%")
m5.metric("PASS QC",f"{(df['QC Status']=='PASS').sum():,}");m6.metric("Needs review",f"{(df['QC Status']!='PASS').sum():,}")

tabs=st.tabs(["🏠 Dashboard","🧾 Sequence Table","📈 Charts","🧹 Research QC","🔬 Sliding Analysis",
              "🧬 ORF & Translation","🧪 Motif Lab","🔢 k-mer & Codon","🧫 Restriction Sites",
              "🧪 Primer Lab","🔎 Compare","📄 Report","⬇️ Export"])

with tabs[0]:
    st.subheader("Dataset Dashboard")
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(px.histogram(df,x="Length",nbins=min(30,max(5,len(df))),title="Sequence Length Distribution"),use_container_width=True)
    with c2: st.plotly_chart(px.histogram(df,x="GC %",nbins=20,title="GC% Distribution"),use_container_width=True)
    summary=pd.DataFrame({"Metric":["Sequences","Total bases","Minimum length","Maximum length","Mean length","Median length","Mean GC%","Mean entropy"],
                          "Value":[len(df),int(df["Length"].sum()),int(df["Length"].min()),int(df["Length"].max()),
                                   round(df["Length"].mean(),2),round(df["Length"].median(),2),round(df["GC %"].mean(),2),round(df["Entropy"].mean(),4)]})
    st.dataframe(summary,use_container_width=True,hide_index=True)
    types=df["Type"].value_counts().rename_axis("Type").reset_index(name="Count")
    st.plotly_chart(px.pie(types,names="Type",values="Count",title="Detected Sequence Types"),use_container_width=True)

with tabs[1]:
    st.subheader("Detailed Sequence Table")
    query=st.text_input("Search ID or description")
    lo,hi=int(df["Length"].min()),int(df["Length"].max())
    filtered=df.copy()
    if lo<hi:
        rng=st.slider("Length range",lo,hi,(lo,hi));filtered=filtered[filtered["Length"].between(*rng)]
    if query:
        mask=filtered["ID"].str.contains(query,case=False,na=False)|filtered["Description"].str.contains(query,case=False,na=False)
        filtered=filtered[mask]
    st.dataframe(filtered,use_container_width=True,hide_index=True)
    sid=st.selectbox("Inspect sequence",list(df["ID"]))
    rec=next(r for r in records if r.id==sid);st.code(clean_sequence(rec.seq),language="text")

with tabs[2]:
    st.subheader("Interactive Visualization Studio")
    c1,c2=st.columns(2)
    with c1:
        fig=px.bar(df,x="ID",y="GC %",title="GC% by Sequence");fig.update_layout(xaxis_tickangle=-45);st.plotly_chart(fig,use_container_width=True)
    with c2: st.plotly_chart(px.scatter(df,x="Length",y="GC %",hover_name="ID",title="Length vs GC%"),use_container_width=True)
    comp=df.melt(id_vars="ID",value_vars=["A","T","G","C"],var_name="Base",value_name="Count")
    st.plotly_chart(px.bar(comp,x="ID",y="Count",color="Base",barmode="group",title="A/T/G/C Counts"),use_container_width=True)
    corr=df.select_dtypes(include=np.number).corr().round(3)
    st.dataframe(corr,use_container_width=True)
    st.plotly_chart(px.imshow(corr,text_auto=True,aspect="auto",title="Feature Correlation Matrix"),use_container_width=True)
    pos=position_composition(records)
    if not pos.empty: st.plotly_chart(px.line(pos,x="Position",y=["A %","T %","G %","C %"],title="Position-wise Base Composition"),use_container_width=True)

with tabs[3]:
    st.subheader("Research Quality Control")
    q1,q2,q3,q4=st.columns(4)
    q1.metric("Mean score",f"{df['QC Score'].mean():.1f}/100");q2.metric("PASS",int((df["QC Status"]=="PASS").sum()))
    q3.metric("REVIEW",int((df["QC Status"]=="REVIEW").sum()));q4.metric("FAIL",int((df["QC Status"]=="FAIL").sum()))
    st.dataframe(df[["ID","Length","Ambiguous bases","Longest homopolymer","Entropy","QC Score","QC Status","QC Flags"]],
                 use_container_width=True,hide_index=True)
    st.plotly_chart(px.bar(df,x="ID",y="QC Score",title="Per-sequence QC Score"),use_container_width=True)
    st.plotly_chart(px.imshow(df.set_index("ID")[["A","T","G","C"]].T,text_auto=True,aspect="auto",title="Nucleotide Count Heatmap"),use_container_width=True)

with tabs[4]:
    st.subheader("Sliding-Window Analysis")
    sid=st.selectbox("Sequence",list(df["ID"]),key="slide")
    seq=clean_sequence(next(r for r in records if r.id==sid).seq)
    if len(seq)<5: st.warning("Sequence is too short.")
    else:
        window=st.number_input("Window size",5,len(seq),min(100,len(seq)),5)
        metric=st.selectbox("Metric",["GC %","GC Skew","AT Skew","Entropy"])
        wdf=sliding_metric(seq,int(window),metric)
        st.plotly_chart(px.line(wdf,x="Position",y=metric,title=f"Sliding {metric} — window {window}"),use_container_width=True)
        st.dataframe(wdf.head(2000),use_container_width=True,hide_index=True)

with tabs[5]:
    st.subheader("ORF Screening & Translation")
    sid=st.selectbox("DNA sequence",list(df["ID"]),key="orf")
    seq=clean_sequence(next(r for r in records if r.id==sid).seq)
    minimum=st.number_input("Minimum ORF length (aa)",0,10000,20,5)
    if set(seq)<=DNA:
        result=longest_orf(seq,int(minimum))
        if result:
            st.success(f"Longest ORF: {result['AA length']} aa, frame +{result['Frame']}, positions {result['Start']}–{result['Stop']}.")
            st.code(result["Protein"],language="text")
            st.dataframe(pd.DataFrame([{k:v for k,v in result.items() if k!="Protein"}]),use_container_width=True,hide_index=True)
        else: st.info("No ORF met the selected threshold.")
        st.dataframe(six_frame_translation(seq),use_container_width=True,hide_index=True)
    else: st.warning("ORF analysis requires A/C/G/T only.")

with tabs[6]:
    st.subheader("Motif & Pattern Laboratory")
    sid=st.selectbox("Sequence",list(df["ID"]),key="motif")
    seq=clean_sequence(next(r for r in records if r.id==sid).seq)
    motifs=st.text_area("Motifs separated by commas","ATG, TATA, GAATTC")
    rows=[]
    for motif in [x.strip().upper() for x in motifs.split(",") if x.strip()]:
        pos=motif_positions(seq,motif)
        rows.append({"Motif":motif,"Length":len(motif),"Hits":len(pos),"First positions":", ".join(map(str,pos[:20])) or "None"})
    mdf=pd.DataFrame(rows)
    if not mdf.empty:
        st.dataframe(mdf,use_container_width=True,hide_index=True)
        st.plotly_chart(px.bar(mdf,x="Motif",y="Hits",title="Motif Abundance"),use_container_width=True)

with tabs[7]:
    st.subheader("k-mer & Codon Analytics")
    sid=st.selectbox("Sequence",list(df["ID"]),key="kmer")
    seq=clean_sequence(next(r for r in records if r.id==sid).seq)
    k=st.selectbox("k-mer size",[2,3,4,5,6],index=1)
    kt=kmer_table(seq,int(k))
    if not kt.empty:
        top=st.slider("Top k-mers",5,min(50,len(kt)),min(20,len(kt)))
        st.plotly_chart(px.bar(kt.head(top),x="k-mer",y="Count",title=f"Top {top} {k}-mers"),use_container_width=True)
        st.dataframe(kt,use_container_width=True,hide_index=True)
    cu=codon_usage(seq)
    if not cu.empty:
        st.plotly_chart(px.bar(cu,x="Codon",y="Count",title="Codon Usage"),use_container_width=True)
        st.dataframe(cu,use_container_width=True,hide_index=True)
    else: st.info("Codon usage requires an unambiguous DNA sequence.")

with tabs[8]:
    st.subheader("Restriction-Site Screening")
    sid=st.selectbox("DNA sequence",list(df["ID"]),key="restriction")
    seq=clean_sequence(next(r for r in records if r.id==sid).seq)
    if set(seq)<=DNA:
        rdf=restriction_table(seq);st.dataframe(rdf,use_container_width=True,hide_index=True)
        st.plotly_chart(px.bar(rdf,x="Enzyme",y="Sites",title="Restriction Sites Detected"),use_container_width=True)
    else: st.warning("Requires A/C/G/T only.")

with tabs[9]:
    st.subheader("Primer / Oligonucleotide Screening")
    sid=st.selectbox("Reference sequence",list(df["ID"]),key="primer")
    seq=clean_sequence(next(r for r in records if r.id==sid).seq)
    oligo=clean_sequence(st.text_input("DNA oligo",seq[:20]))
    gmin,gmax=st.slider("Preferred GC range (%)",0,100,(40,60))
    if oligo:
        if set(oligo)<=DNA:
            a,t,g,c=[oligo.count(x) for x in "ATGC"];gc=100*(g+c)/len(oligo)
            tm=2*(a+t)+4*(g+c);hb,hl=longest_homopolymer(oligo)
            p1,p2,p3,p4=st.columns(4);p1.metric("Length",f"{len(oligo)} nt");p2.metric("GC%",f"{gc:.1f}%")
            p3.metric("Wallace Tm",f"{tm:.1f} °C");p4.metric("GC target","PASS" if gmin<=gc<=gmax else "REVIEW")
            warnings=[]
            if len(oligo)<15 or len(oligo)>35:warnings.append("Length outside common screening range.")
            if not gmin<=gc<=gmax:warnings.append("GC% outside selected range.")
            if hl>=5:warnings.append(f"Homopolymer detected: {hb}{hl}.")
            if warnings:
                for w in warnings:st.warning(w)
            else:st.success("No screening warnings.")
            st.caption("Wallace-rule Tm is an approximate screening value.")
        else:st.error("Use A/C/G/T only.")

with tabs[10]:
    st.subheader("Comparative Sequence Analysis")
    chosen=st.multiselect("Select sequences",list(df["ID"]),default=list(df["ID"])[:min(5,len(df))])
    if chosen:
        cdf=df[df["ID"].isin(chosen)].copy();metrics=["Length","GC %","AT %","GC Skew","AT Skew","Entropy","QC Score"]
        metric=st.selectbox("Comparison metric",metrics)
        st.dataframe(cdf[["ID"]+metrics].round(4),use_container_width=True,hide_index=True)
        fig=px.bar(cdf,x="ID",y=metric,title=f"Comparison — {metric}");fig.update_layout(xaxis_tickangle=-45);st.plotly_chart(fig,use_container_width=True)
        long=cdf.melt(id_vars="ID",value_vars=["A","T","G","C"],var_name="Base",value_name="Count")
        st.plotly_chart(px.bar(long,x="ID",y="Count",color="Base",barmode="group",title="Base Composition Comparison"),use_container_width=True)

with tabs[11]:
    st.subheader("Automated Research Report")
    report=make_report(df);st.text_area("Report preview",report,height=430)
    st.download_button("⬇️ Download report",report.encode(),file_name="multi_fasta_analysis_report.txt",mime="text/plain",use_container_width=True)

with tabs[12]:
    st.subheader("Export Center")
    st.download_button("⬇️ CSV",df.to_csv(index=False).encode(),file_name="sequence_statistics.csv",mime="text/csv",use_container_width=True)
    st.download_button("⬇️ Normalized FASTA",normalized_fasta(records),file_name="normalized_sequences.fasta",mime="text/plain",use_container_width=True)
    xbuf=io.BytesIO()
    with pd.ExcelWriter(xbuf,engine="openpyxl") as writer:
        df.to_excel(writer,sheet_name="Statistics",index=False)
        df[["ID","Length","Ambiguous bases","Entropy","QC Score","QC Status","QC Flags"]].to_excel(writer,sheet_name="QC",index=False)
        df[["ID","A","T","G","C","GC %","AT %","GC Skew","AT Skew"]].to_excel(writer,sheet_name="Composition",index=False)
    st.download_button("⬇️ Excel workbook",xbuf.getvalue(),file_name="multi_fasta_analysis.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)

st.markdown('<div class="muted" style="text-align:center;margin-top:2rem;">Multi-FASTA Sequence Analyzer • Computational screening and research analytics</div>',unsafe_allow_html=True)
