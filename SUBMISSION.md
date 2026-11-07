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

Metric depth models must work out the camera from the pixels to give depth in metres. We test whether they do with a digital zoom: crop an image and resize it back. This test is free and exact. The focal length grows by the zoom factor s, and every pixel that stays in view keeps its depth. Two numbers describe how a model reacts: its depth changes like s^(−α) (ideal α=0) and its predicted focal like s^β (ideal β=1). Neither needs ground truth. Earlier crop tests checked depth maps only; we also check the metric scale and the predicted camera against the true zoomed camera. On 800 images from KITTI, NYUv2 and DIODE, the widely used Depth Anything V2 has α=0.95: it reads depth almost only from how large things look. Models that predict their own camera close only part of the gap (α≈0.31–0.36), and their focal grows too little; Depth Pro lies on the line α+β=1 that size-based reasoning predicts. Given the true camera, Depth Pro and DA3-Nested keep their depth (α≈0), so their focal head carries the whole error. The reason is the classical focal–depth ambiguity, which we state exactly for a zoom: a zoomed photo is, pixel for pixel, a photo of the same scene squashed along the viewing direction, and size cues cannot tell the two apart. Because of this, a focal head trained by regression shrinks towards its training average, and β measures how reliable its camera cue is. Undoing this shrinkage needs no labels, no training and no zoom factor, and it makes Depth Pro equivariant (α=0.04) at almost no cost on unzoomed images. Controls, including a real zoom lens, rule out interpolation and loss of detail, and a cue test shows that the larger size of the content, not the lost context, drives pixels-only models. Finally, self-consistency fine-tuning on unlabeled images lowers α to 0.05 on KITTI and 0.07 on NYUv2 once a blur term removes a "blurry means zoomed" shortcut; we report what this costs on unzoomed and unseen images.

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
