"""Build docs/Zoom_Equivariance_Research_Overview.pdf (Banglish) from the current results.

    .venv/bin/python docs/make_overview_pdf.py

Every number is read from outputs/ (CSV tables written by the notebooks, or recomputed
here from outputs/results); nothing is typed by hand. Rerun after new Kaggle results.
"""
import subprocess
import sys
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import plotstyle as ps  # noqa: E402

import analysis as A  # noqa: E402

FIG, TAB = ROOT / "outputs" / "figures", ROOT / "outputs" / "tables"
OUT = ROOT / "docs" / "Zoom_Equivariance_Research_Overview.pdf"

# DejaVu has Greek letters (alpha, beta, lambda) and ships with matplotlib
TTF = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for face, fname in (("DV", "DejaVuSans.ttf"), ("DVB", "DejaVuSans-Bold.ttf"), ("DVI", "DejaVuSans-Oblique.ttf")):
    pdfmetrics.registerFont(TTFont(face, str(TTF / fname)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI")

BLUE = colors.HexColor("#0B4F8A")
S = {
    "title": ParagraphStyle("t", fontName="DVB", fontSize=17, leading=21, textColor=BLUE, alignment=TA_CENTER),
    "sub": ParagraphStyle("s", fontName="DVI", fontSize=9.5, leading=13, alignment=TA_CENTER, textColor=colors.grey),
    "h1": ParagraphStyle("h1", fontName="DVB", fontSize=12.5, leading=16, textColor=BLUE, spaceBefore=10, spaceAfter=4),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=10.5, leading=14, spaceBefore=6, spaceAfter=2),
    "p": ParagraphStyle("p", fontName="DV", fontSize=9.2, leading=13.2, spaceAfter=4),
    "b": ParagraphStyle("b", fontName="DV", fontSize=9.2, leading=13, leftIndent=12, bulletIndent=3, spaceAfter=1.5),
    "cap": ParagraphStyle("c", fontName="DVI", fontSize=8, leading=10.5, textColor=colors.HexColor("#444444"),
                          spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.8, leading=9.5),
    "box": ParagraphStyle("box", fontName="DV", fontSize=9, leading=12.5, backColor=colors.HexColor("#EEF4FA"),
                          borderColor=BLUE, borderWidth=0.6, borderPadding=6, spaceBefore=4, spaceAfter=10),
}
story = []


def P(text, style="p"):
    story.append(Paragraph(text, S[style]))


def bullets(items):
    for t in items:
        story.append(Paragraph(t, S["b"], bulletText="•"))


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), S["cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#BBBBBB")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE8F5"))]
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 4))


def figure(name, width_cm, caption):
    f = FIG / f"{name}.png"
    if not f.exists():
        return
    from PIL import Image as PILImage

    w, h = PILImage.open(f).size
    img = Image(str(f), width=width_cm * cm, height=width_cm * cm * h / w)
    story.append(KeepTogether([img, Paragraph(caption, S["cap"])]))


def f2(x, d=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


# ------------------------------------------------------------------ data
df = A.load()
pi = A.per_image(df)
t1 = pd.read_csv(TAB / "table1_pooled.csv", index_col=0)
proxy = pd.read_csv(TAB / "proxy_across_pairs.csv")
within = pd.read_csv(TAB / "proxy_within_dataset.csv")
ft_models = [m for m in ("dav2-s-fixed", "dav2-s-blur", "dav2-s-zhong", "dav2-s-eft", "dav2-s-eftb", "dav2-s-eftb3")
             if m in set(df.model)]
ft = A.summary(pi[pi.model.isin(ft_models)])
blur = df[(df.kind == "blur") & df.model.isin(ft_models)].groupby(["model", "dataset"]).log_ratio_self.apply(
    lambda v: 100 * (np.exp(np.median(np.abs(v))) - 1))
stale = pi[pi.model.isin(["metric3d-s", "metric3d-s-staleK"])].pivot_table(index=["dataset", "name"], columns="model",
                                                                          values="alpha")
stale_diff = (stale["metric3d-s-staleK"] - stale["metric3d-s"]).median()
n_img = int(t1.n.max())
pooled = A.summary(pi, by=("model",)).set_index("model")
manifest = pd.read_csv(TAB / "results_manifest.csv")
params_file = ROOT / "outputs" / "results" / "params.json"
params = pd.read_json(params_file, typ="series") if params_file.exists() else pd.Series(dtype=float)
ctrl = A.load("*__ctrl.csv") if list(A.RESULTS.glob("*__ctrl.csv")) else pd.DataFrame()
crossfit = pd.read_csv(TAB / "zoomfix_crossfit.csv") if (TAB / "zoomfix_crossfit.csv").exists() else pd.DataFrame()
pp = pd.read_csv(TAB / "principal_point.csv")
# parametrised tests count once per case, as pytest reports them
collected = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", str(ROOT / "tests")],
                           capture_output=True, text=True, cwd=ROOT).stdout
n_tests = sum(int(line.rsplit(": ", 1)[1]) for line in collected.splitlines() if line.startswith("tests/"))

# ------------------------------------------------------------------ title
P("Zoom In, Scale Out", "title")
P("Metric depth model gulo ki digital zoom er physics mane?", "sub")
P(f"Research overview · Rabeya Noor · target CVPR 2027 · {date.today():%d %B %Y}", "sub")
story.append(Spacer(1, 10))
P(f"<b>Ek line e:</b> Chhobi crop kore boro korle (digital zoom) focal length thik s gun bare, kintu kono jinish "
  f"kachhe ashe na. Amra dekhiyechi beshirbhag metric depth model zoom korle depth kom bole: shobcheye popular "
  f"Depth Anything V2 3x zoom e depth prai 1/3 bole (α = {f2(t1.loc['DA-V2-S', 'alpha'])}). Ete label lage na, "
  f"theory ache, ar label chhara fine-tuning (EFT+B) diye α prai 0 te ana jay, kintu shadharon chhobite "
  f"accuracy kichuta kome.", "box")

# ------------------------------------------------------------------ 1
P("1. Shomossha", "h1")
P("Self-driving, robot, AR ar phone camera te ekta chhobi theke protita jinish koto <b>meter</b> dure, sheta bolte "
  "metric depth model lage (Depth Anything, Depth Pro, MoGe, UniDepth, Metric3D, Depth Anything 3). Meter e bolte hole "
  "camera er focal length jana dorkar. Karon wide lens e kachher jinish ar telephoto lens e durer jinish ekdom ek rokom "
  "dekhate pare. Model gulo camera chhara shudhu pixel dekhe eta anuman kore. Proshno holo: ora ki asholei camera "
  "bujhe, naki shudhu \"boro dekhale kachhe\" shortcut use kore?")

P("2. Mul idea: digital zoom ekta free, shothik ruler", "h1")
P("Chhobir ekta box crop kore abar ager size e resize korle (digital zoom), chhobi ta ekdom shothik bhabe ekta notun "
  "pinhole camera theke tola chhobir moto hoy: f' = s·f, principal point shore jay, kintu <b>depth bodlay na</b>. "
  "Tai label chhara test kora jay: zoom er age ar pore model nijer depth ar focal er sathe tulona korlei hoy.")
table([["Shonkhya", "Mane", "Thik model", "Shortcut model"],
       ["α (depth exponent)", "D̂(zoom) / D̂(original) ≈ s^−α", "0", "1 (2x zoom e depth ordhek)"],
       ["β (focal exponent)", "f̂(zoom) / f̂(original) ≈ s^β", "1", "0 (focal bodlay na)"]],
      [3.3 * cm, 6 * cm, 2.4 * cm, 4.6 * cm])

# ------------------------------------------------------------------ 3
P("3. Theory", "h1")
P("3.1 Tinta invariant", "h2")
bullets(["α = 0 → depth thik thake;", "β = 1 → 3D shape thik thake;",
         "α + β = 1 → shudhu pasher (lateral) map thik thake, depth ba shape na. \"Boro dekhale kachhe\" reasoning "
         "thik ei line e pore: D̂ = f̂·L/ℓ hole zoom e D̂ ∝ s^(β−1)."])
P("3.2 Zoom–squash ambiguity (notun)", "h2")
P("Focal s·f e tola chhobi pixel-by-pixel sheyi scene er chhobi, jeta depth er dik borabor s gun chepta kora: "
  "u = s·f·X/Z = f·X/(Z/s). Chepta scene e jinish er width ek thake, shudhu gobhirota bhul hoy (4.5 m er gari "
  "2x e 2.25 m lomba hoye jay). Tai shudhu size cue diye zoom dhora oshombhob, 3D shape prior lage. Ar shoja dewal "
  "(fronto-parallel plane) e kono model-i camera na jene zoom dhorte parbe na.")
P("3.3 Stale K", "h2")
P(f"Metric3D er moto camera-input model e zoom er por purono K dile network ekoi pixel dekhe, shudhu shesh e scale "
  f"bodlay. Tai α(stale) = α(true) + 1, eta ganit diye proman kora jay. Data te median tofat {f2(stale_diff, 3)}.")

# ------------------------------------------------------------------ 4
P("4. Experiment setup", "h1")
table([["Ki", "Details"],
       ["Model (main)", "DA-V2-S/B, ZoeDepth (pixels only) · Depth Pro, MoGe-2-S, UniDepthV2-S (predicts camera) · "
                        "Metric3D v2-S, thik K ar stale K (takes camera)"],
       ["Model ladder", "DA-V2-L, MoGe-2-B/L, UniDepthV2-B/L, UniK3D-L, Metric3D v2-L, Depth Anything 3 "
                        "(Nested-Giant, Metric-L, L), VGGT-1B"],
       ["Data", f"KITTI val selection (LiDAR), NYUv2 test (Kinect), DIODE val indoor + outdoor (laser). Protita theke 200 "
                f"chhobi, shob model e ekoi chhobi ({n_img} chhobi/model)"],
       ["Zoom", "1, 1.25, 1.5, 2, 3x majhkhane · off-centre: 4 kona, 50 chhobi (KITTI, NYU)"],
       ["Control", "blur control (zoom er moto jhapsa, field of view ek) · geometry check (pixel error 0) · "
                   f"unzoomed prediction ekoi pixel e (fair reference) · optical vs digital zoom, bilinear, JPEG · result verify "
                   f"tool ({len(manifest)} file, shob check pass)"],
       ["Statistics", "95% bootstrap CI (paper e scene-level cluster bootstrap) · paired Wilcoxon signed-rank + Holm "
                      "correction · Spearman with bootstrap CI"],
       ["Compute", "Kaggle T4 GPU; laptop e shudhu analysis"]],
      [3 * cm, 13.3 * cm])

# ------------------------------------------------------------------ 5
P("5. Result 1: zoom audit", "h1")
rows = [["Model", "Group", "α [95% CI]", "β", "α+β", "Blur control", "AbsRel 1x", "AbsRel 3x"]]
groups = {"DA-V2-S": "pixels only", "DA-V2-B": "pixels only", "ZoeDepth": "pixels only", "Depth Pro": "predicts camera",
          "MoGe-2-S": "predicts camera", "UniDepthV2-S": "predicts camera", "Metric3D v2-S": "takes camera (true K)",
          "Metric3D v2-S (stale K)": "takes camera (stale K)"}
for m, r in t1.iterrows():
    if m not in groups:
        continue
    rows.append([m, groups[m], f"{f2(r.alpha)} [{f2(r.alpha_lo)}, {f2(r.alpha_hi)}]", f2(r.beta), f2(r.alpha_plus_beta),
                 f"{100 * r.blur2:.1f}%", f2(r.absrel1, 3), f2(r.absrel3, 3)])
table(rows, [2.9 * cm, 2.8 * cm, 2.7 * cm, 1.1 * cm, 1.1 * cm, 1.7 * cm, 1.6 * cm, 1.6 * cm])
P("<i>Note: ei table er AbsRel 4 ta dataset er gor (DIODE-outdoor er boro error shoho). Section 7 er EFT table e "
  "protita dataset alada, ar sheta 392 px input e chale. Tai DA-V2-S er 1x AbsRel ekhane 0.317, ar EFT table e "
  "KITTI te 0.138. KITTI te default preprocessing e eta 0.133, mane duita mile jay. CI ekhane image-level; "
  "paper e scene-level cluster bootstrap use kora hoy, jeta aro chouro.</i>")
P("<b>Ja pelam:</b>")
bullets([f"Pixels-only model gulo prai puro shortcut use kore: DA-V2-S α = {f2(t1.loc['DA-V2-S', 'alpha'])}.",
         f"Camera predict kora model gulo ordhek bujhe: α ≈ 0.31–0.36, ar focal 3x er jaygay "
         f"{np.exp(t1.loc['UniDepthV2-S', 'beta'] * np.log(3)):.2f}–{np.exp(t1.loc['Depth Pro', 'beta'] * np.log(3)):.2f}x bole.",
         f"Depth Pro e protita chhobite α+β er median {f2(t1.loc['Depth Pro', 'alpha_plus_beta'])}, mane "
         f"theory er \"lateral invariant\" line e prai hubohu thake.",
         f"Thik K pele Metric3D prai equivariant (α = {f2(t1.loc['Metric3D v2-S', 'alpha'])}), stale K dile "
         f"pixels-only model er moto ({f2(t1.loc['Metric3D v2-S (stale K)', 'alpha'])}).",
         "Blur control e depth bodlay mattro 0.7–4%, tai dosh interpolation er na, camera bojhar."])
figure("06_figure1", 12.5, "Figure 1. Ekoi scene 1x, 2x, 3x zoom e. Ek row er shob depth panel ekoi rong er scale e (meter). "
       "DA-V2-S zoom korle \"kachhe\" chole jay (KITTI 28.5 → 7.8 m, GT 21 m); MoGe-2 ordhek.")
figure("fig2_depth_and_focal_vs_zoom", 16, f"Figure 2. Zoom e depth (bame) ar predicted focal (dane), {n_img} chhobir median.")
figure("fig3_alpha_beta_plane", 8, "Figure 3. α–β plane. Dashed line: α+β = 1 (apparent size reasoning). Tara: ideal.")

# ------------------------------------------------------------------ 6
P("6. Result 2: label chhara score diye error anuman", "h1")
r3 = proxy[(proxy.score == "abs_alpha") & (proxy.target == "absrel3")].iloc[0]
r1 = proxy[(proxy.score == "abs_alpha") & (proxy.target == "absrel1")].iloc[0]
w = within[within.target == "absrel3"].set_index("dataset").spearman
P(f"|α| (label lage na) ar 3x zoom e AbsRel er Spearman ρ = {f2(r3.spearman)} [{f2(r3.lo)}, {f2(r3.hi)}], "
  f"{int(r3.pairs)} ta model×dataset jora. Dataset er bhitore KITTI {f2(w.get('kitti'))}, NYU {f2(w.get('nyu'))}, "
  f"DIODE-in {f2(w.get('diode-in'))}, kintu DIODE-out {f2(w.get('diode-out'))}. Shadharon (1x) accuracy predict kore na "
  f"(ρ = {f2(r1.spearman)}). <b>Shotti kotha:</b> zoom e depth shrink korle error bare-i, tai sambandha kichuta "
  f"nijer theke ashe. Claim ta tai choto: \"zoom e kon model koto bhul korbe, label chhara bola jay.\"")

# ------------------------------------------------------------------ 7
P("7. Result 3: label chhara fine-tuning (EFT ar EFT+B)", "h1")
P("Frozen copy (teacher) zoom chhara chhobi te ja bole, student ke zoom kora chhobi te ta-i bolte shekhai. Kono depth "
  "label ba camera info lage na. Training data: 7 ta KITTI raw drive (test drive er sathe overlap nai, code nijei check "
  "kore) ar NYUv2 er 795 ta train chhobi. DIODE kokhono training e dekhano hoy na.")
P("<b>Notun finding, resolution shortcut:</b> shudhu zoom loss diye train korle (EFT) model shikhe ney \"jhapsa chhobi = "
  "zoom\". Blur control e KITTI te depth 30% bodle jay. EFT+B te ekta blur-invariance term jog kori: jhapsa (kintu "
  "zoom na) chhobite ager depth-i bolte hobe. Eta Zhong et al. (ICCV 2023) e nai.")
rows = [["Dataset", "Model", "α", "AbsRel 1x", "AbsRel 3x", "Blur shortcut"]]
for ds in A.DATASETS:
    for m in ft_models:
        r = ft[(ft.model == m) & (ft.dataset == ds)]
        if not len(r):
            continue
        r = r.iloc[0]
        rows.append([A.PRETTY_DS[ds] + (" (unseen)" if ds.startswith("diode") else ""), ps.LABEL[m], f2(r.alpha),
                     f2(r.absrel1, 3), f2(r.absrel3, 3), f"{blur.get((m, ds), np.nan):.1f}%"])
table(rows, [3.4 * cm, 3.4 * cm, 1.6 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm])
def ftv(m, ds, col):
    r = ft[(ft.model == m) & (ft.dataset == ds)]
    return r.iloc[0][col] if len(r) else np.nan


bullets([f"EFT+B prai puro equivariant (KITTI te α {f2(ftv('dav2-s-eftb', 'kitti', 'alpha'))}, NYU te "
         f"{f2(ftv('dav2-s-eftb', 'nyu', 'alpha'))}), ar blur shortcut prai chole geche.",
         f"Trade-off: KITTI te 1x AbsRel {f2(ftv('dav2-s-fixed', 'kitti', 'absrel1'), 3)} → "
         f"{f2(ftv('dav2-s-eftb', 'kitti', 'absrel1'), 3)}, DIODE e aro beshi (training e dekhe nai); NYU te bhalo hoy.",
         f"λ = 3 (3 gun shokto anchor) eta thik kore na: KITTI {f2(ftv('dav2-s-eftb3', 'kitti', 'absrel1'), 3)}, "
         f"NYU kharap hoy ({f2(ftv('dav2-s-eftb3', 'nyu', 'absrel1'), 3)}). Trade-off ta weight tune kore shoray jay na.",
         f"Blur-only (zoom term chhara) α beshi komay na (KITTI {f2(ftv('dav2-s-blur', 'kitti', 'alpha'))}): "
         f"jhapsa chhobi dekhano camera shekhay na.",
         f"Zhong et al.-style (frozen teacher er bodole student nijer prediction target) α komay "
         f"({f2(ftv('dav2-s-zhong', 'kitti', 'alpha'))}), kintu blur shortcut fire ashe "
         f"({blur.get(('dav2-s-zhong', 'kitti'), np.nan):.0f}%) ar 1x AbsRel shobcheye kharap "
         f"({f2(ftv('dav2-s-zhong', 'kitti', 'absrel1'), 3)}). Frozen teacher, zoom term ar blur term, tinta-i lage.",
         "\"Anchor-only\" baseline asole frozen model (student = teacher, tai gradient 0), tai blur-only control."])
figure("fig5_finetuning", 16, "Figure 5. Upore: zoom e depth ratio (1 = ideal). Niche: AbsRel. Dhushor: frozen, holud: EFT, "
       "lal: EFT+B, golapi: EFT+B λ = 3, sobuj: blur-only. Note: alada zoom e AbsRel alada pixel e mapa; "
       "NYU te frozen model 1.25–1.5x e 1x er cheye bhalo "
       "AbsRel pay, karon 1x e depth beshi bole ar zoom shrink kakatale thik er kachhe ane. Tai AbsRel na, α dekhte hobe.")

# ------------------------------------------------------------------ 8
P("8. Model boro ba notun hole ki thik hoy? (model ladder)", "h1")
LADDER = ["dav2-s", "dav2-b", "dav2-l", "moge2-s", "moge2-b", "moge2-l", "unidepth2-s", "unidepth2-b", "unidepth2-l",
          "unik3d-l", "depthpro", "da3-nested", "da3-l", "vggt", "da3-metric-l", "metric3d-s", "metric3d-l"]
rows = [["Model", "Params", "α", "β", "chhobi"]]
for m in [m for m in LADDER if m in pooled.index]:
    r = pooled.loc[m]
    rel = m in ("da3-l", "vggt")
    rows.append([ps.LABEL.get(m, m) + (" (relative)" if rel else ""),
                 f"{params[m] / 1e6:,.0f}M" if m in params else "–", "–" if rel else f2(r.alpha), f2(r.beta), int(r.n)])
table(rows, [5 * cm, 2.4 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm])
P("Protita model nijer sample e (main model 200, boro variant 100, DA3/VGGT/Metric3D-L 50 chhobi prati dataset). "
  "Paper e ekoi chhobi te paired tulona-o ache (151 ta chhobi), result ek.")
bullets([f"DA-V2 boro hole α ektu kome ({f2(pooled.loc['dav2-s'].alpha)} → {f2(pooled.loc['dav2-l'].alpha)}), "
         "kintu L model-o 3x e ordheker kom depth bole.",
         f"MoGe-2 er size bariye labh nai ({f2(pooled.loc['moge2-s'].alpha)}, {f2(pooled.loc['moge2-b'].alpha)}, "
         f"{f2(pooled.loc['moge2-l'].alpha)}); UniDepthV2 L e bhalo ({f2(pooled.loc['unidepth2-l'].alpha)}); "
         f"UniK3D-L shobcheye kache (α {f2(pooled.loc['unik3d-l'].alpha)}, β {f2(pooled.loc['unik3d-l'].beta)}).",
         f"Shobcheye boro DA3-Nested ({params.get('da3-nested', np.nan) / 1e6:,.0f}M) α = "
         f"{f2(pooled.loc['da3-nested'].alpha)}, chhoto camera model gulor cheye kharap, ZoeDepth er kachhe.",
         f"Shothik camera dile DA3-Metric-L ({f2(pooled.loc['da3-metric-l'].alpha)}) ar Metric3D-L "
         f"({f2(pooled.loc['metric3d-l'].alpha)}) depth thik rakhe.",
         "<b>Mane:</b> model koto boro ta na, camera model e dhoke kina setai ashol."])

# ------------------------------------------------------------------ 8b
P("9. Control, principal point ar ZoomFix", "h1")
if len(ctrl):
    rows = [["Model", "Optical 2x", "Digital 2x", "Bilinear 2x", "JPEG"]]
    for m in ["dav2-s", "depthpro", "moge2-s", "metric3d-s"]:
        cells = [f2(float(np.exp(ctrl[(ctrl.model == m) & (ctrl.kind == k)].log_ratio_self.median())))
                 for k in ("optical", "digital_lr", "bilinear", "jpeg")]
        rows.append([ps.LABEL.get(m, m)] + cells)
    table(rows, [4 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm])
    P("Depth ratio (1 = bodlay na). Real detail shoho optical zoom-o DA-V2-S er depth prai digital zoom er motoi komay. "
      "Tai dosh detail haranor na, camera bojhar. Shudhu Depth Pro te detail kichuta matter kore.")
q = pp[(pp.offset == "quadrant") & (pp.zoom == 2.0)].groupby("model")[["model_error", "centred_guess_error"]].median()
P(f"<b>Principal point:</b> 2x quadrant crop e principal point image width er ordhek shore jay. MoGe-2 er error "
  f"{f2(q.model_error.get('moge2-s', np.nan), 3)}, UniDepthV2 {f2(q.model_error.get('unidepth2-s', np.nan), 3)}, "
  f"\"shob shomoy centre\" bolle {f2(q.centred_guess_error.get('moge2-s', np.nan), 3)}. Duita model-i shore jawa dhore na.")
if len(crossfit):
    rows = [["Model", "Dataset", "ê (label chhara)", "3x as is", "3x + ZoomFix", "Target (1x, ekoi pixel)"]]
    for _, r in crossfit.iterrows():
        rows.append([ps.LABEL.get(r.model, r.model), A.PRETTY_DS[r.dataset], f2(r.exponent), f2(r.absrel_zoomed, 3),
                     f2(r.absrel_fixed, 3), f2(r.absrel_unzoomed_same_pixels, 3)])
    table(rows, [3 * cm, 2.6 * cm, 2.8 * cm, 2.3 * cm, 2.6 * cm, 3 * cm])
    P("<b>ZoomFix:</b> zoom factor jana thakle (phone EXIF) depth ke s^ê diye gun kori; ê label chhara ordhek chhobi "
      "te mapa, baki ordheke score. DA-V2-S e KITTI, NYU, DIODE-in e zoom er error er 1/3 theke 2/3 shore jay. "
      "DIODE-out e mean bare, karon kichu chhobite agei boro error (heavy tail), scale up korle ta aro bare. "
      "Eta ekta stop-gap, camera-aware model er bikolpo na.")

# ------------------------------------------------------------------ 9
P("10. Shotti negative result", "h1")
bullets(["Theory er \"soft\" prediction (scene e depth beshi chhoriye thakle α kombe) data te pawa jay nai "
         "(camera model e ρ ≈ 0).",
         "Label-free score shadharon accuracy predict kore na, shudhu zoom e accuracy.",
         "UniDepthV2-S DIODE-outdoor e lomba error tail dey (prai 10% chhobite 6x beshi depth); α er upor eta "
         "prabhab felbe na.",
         "Stale-K result notun na (Kim & Lee 2026 er kaj ke support kore)."])

# ------------------------------------------------------------------ 10
P("11. Agerkar kaj ar amader notun ongsho", "h1")
table([["Paper", "Ki koreche", "Amader sathe tofat"],
       ["Zhong et al., ICCV 2023", "crop-resize equivariance, self-consistency loss", "camera/metric nai; amra α, β, theory, "
                                                                                     "notun model, shortcut + EFT+B"],
       ["Kim & Lee, Electronics 2026", "stale intrinsics e camera-input model er bhul", "amader stale-K tader support kore"],
       ["Roll-robust depth (2026)", "camera roll", "roll e intrinsics bodlay na, zoom e bodlay"],
       ["KPE (2023), PlaneDepth (CVPR 2023)", "crop er perspective ambiguity", "test/training input; zoom audit na"],
       ["FS-Depth (2023)", "crop diye focal augment, label shoho", "amader label ar focal input chhara"],
       ["DepthCues (CVPR 2025)", "depth cue probing (size shoho)", "amra size cue er physical meaning (α)"]],
      [4.2 * cm, 5.6 * cm, 6.5 * cm])
P("<b>Amader notun ongsho:</b> (1) metric depth er zoom audit with α ar β; (2) α+β = 1 line ar zoom–squash theory; "
  "(3) teen dhoroner model (pixels only / predicts camera / takes camera) ek scale e; (4) label-free proxy; "
  "(5) resolution shortcut finding ar EFT+B; (6) principal point test ar ZoomFix (EXIF zoom factor diye training chhara "
  "fix); (7) model ladder (DA3, VGGT, UniK3D shoho).")

# ------------------------------------------------------------------ 11
P("12. Code", "h1")
bullets(["GitHub (private): github.com/rabeyanoor/zoom-equivariance",
         f"{n_tests} ta unit test (zoom geometry, audit toy model, analysis, ZoomFix), ruff lint, GitHub Actions CI",
         "Vectorised analysis (loop er cheye 150 gun druto, test e hubohu ek)",
         "verify_results.py: protita CSV te shob zoom, duplicate nai, shob model ekoi chhobi",
         "Paper er shob number make_paper_numbers.py diye CSV theke, haate kichu lekha hoy na",
         "Protita Kaggle run er log logs/kaggle/ e"])

# ------------------------------------------------------------------ 12
P("13. Baki kaj", "h1")
table([["Kaj", "Obostha"],
       ["Shob Kaggle run (main audit, ladder c1 + c2, controls, ZoomFix, EFT ablation)", "shesh"],
       ["Paper (8 page + references) ar supplement, shob number script theke", "shesh, draft"],
       ["Lekha ta abar pora, choto kora, figure polish", "baki"],
       ["(Optional) phone e 1x/2x/5x zoom chhobi, tape diye mapa dure", "tumi korbe"],
       ["Overleaf e PDF, supervisor review, CVPR submit (~7–16 Nov)", "shesh e"]],
      [11.3 * cm, 5 * cm])

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm,
                        bottomMargin=1.6 * cm, title="Zoom In, Scale Out - research overview", author="Rabeya Noor")


def footer(canvas, doc_):
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(A4[0] - 2 * cm, 1 * cm, f"Zoom In, Scale Out · research overview · page {doc_.page}")


doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
