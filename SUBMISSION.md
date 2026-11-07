# CVPR 2027 submission sheet

Venue: https://openreview.net/group?id=thecvf.com/CVPR/2027/Conference
Deadlines (cvpr.thecvf.com/Conferences/2027/CallForPapers):

- Enrollment (title, abstract, authors): **Nov 10, 2026, 11:59 PM AoE**.
- Full paper: **Nov 16, 2026, 11:59 PM AoE**.

Every author needs an OpenReview profile by Nov 10.

## Title

Zoom In, Scale Out: Metric Depth Models Mistake Digital Zoom for Moving Closer

## Authors (in this order)

1. Rabeya Noor. Dept. of CSE, Daffodil International University. rabeyanoor.dev@gmail.com. ORCID 0009-0005-1045-5927.
2. Khandoker Nosiba Arifin. Dept. of CSE, Daffodil International University. nosibaarifin.cse@diu.edu.bd.

## Abstract (plain text, generated from paper/sec/0_abstract.tex by scripts/abstract_text.py)

Metric depth models must infer the camera from pixels to output metres. We use a digital zoom, a crop followed by a resize, as a free and exact test of whether they do: the focal length is multiplied by the zoom factor s and the depth of every surviving pixel is unchanged. Two exponents summarise a model's response: its depth scales like s^(−α) (ideal α=0) and its predicted focal like s^β (ideal β=1). Neither needs ground truth. Unlike earlier crop-equivariance tests of depth maps, this audit scores the metric scale and the predicted camera against the exact zoomed camera. Across 800 images from KITTI, NYUv2 and DIODE, the widely used Depth Anything V2 has α=0.95, close to reading depth from apparent size alone. Models that predict their own camera close only part of the gap (α≈0.31–0.36) and under-scale the focal; Depth Pro lies on the line α+β=1 that apparent-size reasoning predicts. Given the true camera, Depth Pro and DA3-Nested become equivariant (α≈0): their focal head carries the whole error. We explain this with the classical focal–depth ambiguity, written as an exact zoom–squash statement: a zoomed photo is pixel-for-pixel the photo of the same scene compressed along the viewing axis, which size cues cannot detect. Under it, a regression-trained focal head shrinks towards a prior and β measures the reliability of its camera cue; inverting this shrinkage without labels, training or the zoom factor makes Depth Pro equivariant (α=0.04) at almost no cost on unzoomed images. Controls, including a real zoom lens, rule out interpolation and detail loss, and a cue intervention shows that magnification, not lost context, drives pixels-only models. Finally, self-consistency fine-tuning on unlabeled images reduces α to 0.05 on KITTI and 0.07 on NYUv2 once a blur-invariance term removes a "blurry means zoomed" shortcut; we report its cost on unzoomed and unseen images.

## Keywords

monocular metric depth estimation; camera intrinsics; focal length estimation; equivariance; digital zoom;
label-free evaluation; self-supervised fine-tuning

## TL;DR (if the form asks for one)

A crop-and-resize is an exact digital zoom. Metric depth models read it as moving closer, and for models with
a focal head the whole error sits in that head.

## Files

- `paper/main.pdf`: anonymous submission. 8 pages of main text; then the LLM Usage Statement and references,
  which do not count toward the page limit.
- `paper/supp.pdf`: supplementary material.
- `paper/main_named.pdf`: copy with author names. Not for review; use it for the camera-ready version or arXiv.

## Before submitting

- Do not link this repository from the paper or supplement (double-blind).
- Keep the LLM Usage Statement in the paper. The guidelines make it mandatory.
- Authors are expected to serve as reviewers. Keep both OpenReview profiles complete (affiliation, emails,
  DBLP/ORCID where available).
