"""Octo (baseline) architecture diagram — for side-by-side comparison with FNO-VLA."""
import os
os.environ["PATH"] = r"C:\Users\islab\anaconda3\envs\mmdetection\Library\bin" + os.pathsep + os.environ.get("PATH", "")
from graphviz import Digraph

OUT = r"c:\sarvik\fno_backup\ppt_figures\architecture_octo"
INK = "#1b2a41"
C = {"in":("#8896a6","#eef1f4"), "tok":("#6b7b8c","#e7ebef"), "lm":("#2e6fae","#dde9f4"),
     "bb":("#7d5ba6","#ece3f2"), "head":("#c0392b","#fbe1dd"), "out":("#8896a6","#eef1f4")}

def lbl(title, sub, tsize=15, ssize=10):
    sub = sub.replace("&", "&amp;")
    return (f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3">'
            f'<TR><TD><FONT POINT-SIZE="{tsize}"><B>{title}</B></FONT></TD></TR>'
            f'<TR><TD><FONT POINT-SIZE="{ssize}" COLOR="#3a4a5a">{sub}</FONT></TD></TR></TABLE>>')

g = Digraph("Octo", format="png")
g.attr(rankdir="TB", bgcolor="white", fontname="Helvetica", splines="spline",
       nodesep="0.45", ranksep="0.62", pad="0.4", dpi="300")
g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", color=INK,
       penwidth="1.8", margin="0.18,0.10")
g.attr("edge", color="#7f8c8d", penwidth="1.6", arrowsize="0.9")
g.attr(label=("<<FONT POINT-SIZE='20'><B>Octo</B></FONT>"
              "<FONT POINT-SIZE='13'>  —  baseline (NOT our model) · transformer policy + diffusion action head</FONT><BR/>"
              "<FONT POINT-SIZE='11' COLOR='#8896a6'>Octo-Small 27M / Octo-Base 93M · pretrained on ~800k trajectories (Open X-Embodiment)</FONT>>"),
       labelloc="t", fontname="Helvetica")

def node(nid, grp, title, sub, dashed=False):
    ec, fc = C[grp]
    g.node(nid, lbl(title, sub), fillcolor=fc, color=ec,
           style="rounded,filled,dashed" if dashed else "rounded,filled")

with g.subgraph(name="cluster_in") as c:
    c.attr(label="INPUTS", labeljust="l", fontsize="11", fontcolor="#8896a6", color="#d8dee4", style="rounded", margin="10")
    node("lang","in","Language Task","“pick up the soup …”")
    node("img","in","Observation Images","RGB (+ optional wrist / history)")
    node("prop","in","Proprioception","robot state (optional)")

with g.subgraph(name="cluster_tok") as c:
    c.attr(label="TOKENIZERS", labeljust="l", fontsize="11", fontcolor="#8896a6", color="#d8dee4", style="rounded", margin="10")
    node("t5","lm","T5 Text Encoder","frozen pretrained LM → task tokens")
    node("cnn","tok","Shallow CNN Tokenizer","image patches → observation tokens")
    node("read","tok","Readout Tokens","learnable query tokens")

node("bb","bb","Octo Transformer","block-wise causal attention backbone · multi-modal token sequence")
node("head","head","Diffusion Action Head","iterative denoising → continuous actions")
node("out","out","Action Chunk","predicted delta end-effector + gripper")

g.edge("lang","t5"); g.edge("img","cnn"); g.edge("prop","cnn", style="dashed")
for t in ("t5","cnn","read"): g.edge(t,"bb")
g.edge("bb","head", label="  readout features", fontsize="10", fontcolor=INK, color=INK, penwidth="2.0")
g.edge("head","out", penwidth="2.0", color=INK)

# contrast note
with g.subgraph() as s:
    s.attr(rank="same"); s.node("head")
    s.node("note", shape="note", style="filled", fillcolor="#fff5f4", color="#e0a59e",
           label=("<<TABLE BORDER='0' CELLBORDER='0'><TR><TD ALIGN='LEFT'><FONT POINT-SIZE='9.5' COLOR='#7a2e26'>"
                  "<B>vs. FNO-VLA:</B> Octo uses a heavy<BR ALIGN='LEFT'/>transformer + diffusion head, no<BR ALIGN='LEFT'/>"
                  "scattering, no Fourier head, and is<BR ALIGN='LEFT'/>NOT resolution-invariant.</FONT></TD></TR></TABLE>>"))
g.edge("head","note", style="invis", constraint="false")

g.render(OUT, format="png", cleanup=True)
g.render(OUT, format="svg", cleanup=True)
print("wrote", OUT + ".png and .svg")
