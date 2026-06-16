"""Extremely high-quality FNO-VLA architecture diagram via Graphviz (PNG @300dpi + SVG)."""
import os
os.environ["PATH"] = r"C:\Users\islab\anaconda3\envs\mmdetection\Library\bin" + os.pathsep + os.environ.get("PATH", "")
from graphviz import Digraph

OUT = r"c:\sarvik\fno_backup\ppt_figures\architecture_gv"

# muted, premium palette (border, fill)
C = {
 "in":   ("#94a3b8", "#f3f6f9"),
 "scat": ("#3a9188", "#e7f1ef"),
 "dino": ("#43698f", "#e9eff5"),
 "lang": ("#6f6597", "#eeebf3"),
 "prop": ("#94a3b8", "#f3f6f9"),
 "fuse": ("#b9954f", "#f6efe1"),
 "fno":  ("#b56a4a", "#f4e8e2"),
 "aux":  ("#a3afba", "#f2f5f7"),
 "out":  ("#5e8b71", "#e8f0eb"),
}
INK = "#26313f"

def lbl(title, sub, tsize=15, ssize=10):
    sub = sub.replace("&", "&amp;")
    return (f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">'
            f'<TR><TD><FONT POINT-SIZE="{tsize}" COLOR="{INK}"><B>{title}</B></FONT></TD></TR>'
            f'<TR><TD><FONT POINT-SIZE="{ssize}" COLOR="#5b6878">{sub}</FONT></TD></TR>'
            f'</TABLE>>')

g = Digraph("FNO_VLA", format="png")
g.attr(rankdir="TB", bgcolor="white", fontname="Helvetica", splines="ortho",
       nodesep="0.6", ranksep="0.95", pad="0.5", dpi="300")
g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica",
       color=INK, penwidth="1.8", margin="0.24,0.14")
g.attr("edge", color="#7f8c8d", penwidth="1.6", arrowsize="0.9")

# title
g.attr(label=("<<FONT POINT-SIZE='20'><B>FNO-VLA</B></FONT>"
              "<FONT POINT-SIZE='13'>  —  Scattering + DINOv3 vision · FiLM fusion · "
              "Fourier-Neural-Operator action head</FONT><BR/>"
              "<FONT POINT-SIZE='11' COLOR='#8896a6'>28.9M parameters · RGB-only · LIBERO manipulation</FONT>>"),
       labelloc="t", fontname="Helvetica")

def node(nid, grp, title, sub, dashed=False, emph=False, novel=False):
    ec, fc = C[grp]
    style = "rounded,filled,dashed" if dashed else "rounded,filled"
    g.node(nid, lbl(title, sub, tsize=16 if emph else 15),
           fillcolor=fc, color=ec, style=style, penwidth="2.3" if emph else "1.7")

# ---- inputs ----
with g.subgraph(name="cluster_in") as c:
    c.attr(label="INPUTS", labeljust="l", fontsize="11", fontcolor="#8896a6",
           color="#d8dee4", style="rounded", penwidth="1.0", margin="10")
    node("cams", "in", "Dual RGB Cameras", "agentview + wrist · 128×128 · 6-ch")
    node("lang", "in", "Language", "“pick up the soup …”")
    node("prop", "in", "Proprioception", "15-D · ee_pos, ori, grip, joints")

# ---- encoders ----
with g.subgraph(name="cluster_enc") as c:
    c.attr(label="ENCODERS", labeljust="l", fontsize="11", fontcolor="#8896a6",
           color="#d8dee4", style="rounded", penwidth="1.0", margin="10")
    node("scat", "scat", "Wavelet Scattering", "kymatio J=3, L=12<BR/>+ CNN proj (128, 3 res-blocks)", novel=True)
    node("dino", "dino", "DINOv3 ViT-S/16", "<B>frozen</B> · 21.6M · semantic")
    node("lenc", "lang", "Language Encoder", "learned 2-layer Transformer · dim 128")
    node("penc", "prop", "Proprio MLP", "15 → 64 → 128")

# ---- fusion ----
node("fuse", "fuse", "Multimodal Fusion Transformer",
     "3 layers · 8 heads · dim 256  |  proprio-gated attention pool over views  |  FiLM language conditioning")

# ---- heads ----
node("fno", "fno", "FNO Action Decoder",
     "Fourier Neural Operator · width 256 · 4 layers · <B>8 modes</B> · spectral conv + iFFT", emph=True, novel=True)
node("aux", "aux", "Aux x-y Head", "Linear 256→128 → GELU → 128→2  ·  train-only", dashed=True, novel=True)

# ---- outputs ----
node("act", "out", "Action Trajectory + Gripper",
     "chunk 16 × 6-DoF (Δx Δy Δz Δr Δp Δyaw) + gripper · execute 8/16 @ 20 Hz")
node("grasp", "out", "Predicted grasp (x, y)", "used only in training", dashed=True)

# ---- edges ----
g.edge("cams", "scat"); g.edge("cams", "dino")
g.edge("lang", "lenc"); g.edge("prop", "penc")
for e in ("scat", "dino", "lenc", "penc"):
    g.edge(e, "fuse")
g.edge("fuse", "fno", label="  latent z (256-D)", fontsize="10", fontcolor=INK, fontname="Helvetica", penwidth="2.0", color=INK)
g.edge("fuse", "aux", style="dashed", color="#9aa7b4")
g.edge("fno", "act", penwidth="2.0", color=INK)
g.edge("aux", "grasp", style="dashed", color="#9aa7b4")

# ---- side explanation notes (one per stage, far right, sticky-note style) ----
def note(nid, html):
    g.node(nid, ("<<TABLE BORDER='0' CELLBORDER='0' CELLPADDING='2'><TR><TD ALIGN='LEFT'>"
                 f"<FONT POINT-SIZE='9.5' COLOR='#5a4b1a'>{html}</FONT></TD></TR></TABLE>>"),
           shape="note", style="filled", fillcolor="#fffbe6", color="#d9b94e", penwidth="1.3")
note("n_in",  "<B>Inputs:</B> 2 RGB views + language<BR ALIGN='LEFT'/>+ 15-D robot state (no depth /<BR ALIGN='LEFT'/>privileged info)")
note("n_enc", "<B>Encoders:</B> scattering &#8594; noise-<BR ALIGN='LEFT'/>robust features; DINOv3 (frozen)<BR ALIGN='LEFT'/>&#8594; semantic object features")
note("n_fuse","<B>Fusion + FiLM:</B> modalities<BR ALIGN='LEFT'/>combined; the instruction scales/<BR ALIGN='LEFT'/>shifts features to the named object")
note("n_head","<B>Heads:</B> FNO &#8594; smooth, band-<BR ALIGN='LEFT'/>limited actions at any control rate;<BR ALIGN='LEFT'/>aux x-y head sharpens grasping<BR ALIGN='LEFT'/>(training-only, removed at test)")
note("n_out", "<B>Output:</B> 16-step 6-DoF chunk<BR ALIGN='LEFT'/>+ gripper; execute 8 @ 20 Hz,<BR ALIGN='LEFT'/>then re-plan")
for anchor, n in [("prop","n_in"),("penc","n_enc"),("fuse","n_fuse"),("aux","n_head"),("grasp","n_out")]:
    with g.subgraph() as s:
        s.attr(rank="same"); s.node(anchor); s.node(n)
    g.edge(anchor, n, style="invis", constraint="false")

# legend
with g.subgraph(name="cluster_legend") as c:
    c.attr(label="", color="white")
    c.node("leg", shape="box", style="rounded,filled", fillcolor="#f7f9fb", color="#dde3e9",
           label=("<<TABLE BORDER='0' CELLBORDER='0' CELLPADDING='3'><TR><TD ALIGN='LEFT'>"
                  "<FONT POINT-SIZE='9' COLOR='#5b6878'><B>frozen</B> = pretrained, not updated&nbsp;&nbsp;&nbsp;"
                  "—— train + inference&nbsp;&nbsp;&nbsp;- - - training only (removed at deployment)</FONT>"
                  "</TD></TR></TABLE>>"))

g.render(OUT, format="png", cleanup=True)
g.render(OUT, format="svg", cleanup=True)
print("wrote", OUT + ".png  and  .svg")
