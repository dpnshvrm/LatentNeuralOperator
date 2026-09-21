# Latent Twin Operator — ICLR 2027: status & TODO (2026-09-08)

**Deadlines:** abstract Sept 18, full paper Sept 25.

## Theory (Section 2) — essentially locked

- Definition 2.1 (LTO), Assumption 2.2 (Regularity), Theorem 2.3 (explicit discretization error, O(δ(c_ctx)^(2/3))), Corollary 2.4 (discretization invariance) — all stated, numbered, and stable. Appendix has the deferred proofs (Lemma A.1/Corollary A.2 encoder stability, Proposition A.3 quadrature, Corollary A.4 bandwidth).
- The theorem covers only the encode/decode (SetConv) steps — it's agnostic to how time-evolution (`propagate()`) is implemented, so using a CNN rather than attention there doesn't affect validity. This is also the paper's core novelty argument: classical kernel-smoothing theory exists for SetConv encode/decode; no analogous discretization-rate result exists yet for attention-based latent operators (checked directly against the two closest theory-adjacent papers).
- **Open:** one sentence needed in Limitations — Assumption 2.2's smoothness requirement is stressed (not violated outright, but a real scope limit) by Burgers nu=0.001's shock-forming dynamics. Not yet written.
- **Open:** decide whether to add two theory-adjacent citations (Calvello et al.; Zhang, Leung & Schaeffer) to the Related Work discretization-invariance paragraph.

## Related Work — written, five paragraphs, done

Covers neural operators, implicit neural representations, conditional neural processes/SetConv, latent-space operator learning (LNO/PhCA, Latent Neural PDE Solver, PI-Latent-NO, our own Latent Twins papers), and discretization-invariance critiques (Gao et al. ICLR 2025, Lanthaler-Stuart-Trautner). All citations verified via direct fetch, not from memory.

## Experiments — per PDE track

| Track | Status |
|---|---|
| 2D Heat | Solid. Working notebook, ablations, resolution-transfer sweep, real MSE numbers already quoted in the paper. |
| Darcy | `LNO_Darcy` and `LTO_Darcy` training in parallel on Palmetto (harness-matched comparison). Resolution-transfer eval pipeline built, ready once checkpoints land. |
| 1D Advection (PDEBench) | Real data, trained and evaluated. Done. |
| NS2d | Primary checkpoint now uses the AE reconstruction loss (with rollout self-conditioning kept on): **1.61x / 1.93x to LNO** (non-chained / chained), improving on the earlier paused 1.73x / 2.08x checkpoint. |
| 1D Burgers, nu=0.001 (PDEBench) | Primary checkpoint (no reconstruction loss) **ties FNO** on the head-to-head chained-rollout comparison (0.0785 vs 0.0781 LTO vs FNO), with a genuine, self-trained FNO baseline (no usable published PDEBench number at this viscosity) and an informative short-horizon-win/long-horizon-loss crossover. Supplemented by a full generalization-axis ablation (see below). |
| SWE / earthquake | Likely cut. |

## New this week: Burgers generalization-axis ablation

Investigated whether adding a reconstruction-consistency loss (`decode(encode(y)) ≈ y`) helps or hurts, and how it interacts with the self-conditioning rollout curriculum. Headline finding: **no single training recipe wins on every axis** —

- Fine-grained (1-frame) chaining: rollout is decisively necessary.
- Coarse-grained (5-frame) chaining — the actual FNO-comparison regime: no reconstruction loss at all wins outright; rollout is mildly *counterproductive* here.
- Spatial resolution transfer (evaluating at resolutions the model never trained at): the reconstruction loss *without* rollout wins by a wide margin — roughly halves the error at coarse resolutions vs. either rollout-trained checkpoint.

Read as: the reconstruction loss and the rollout curriculum each buy generalization along different axes and don't compose additively — a genuinely reviewable ablation/mechanism finding, not just a hyperparameter sweep. Full numbers in the project's internal notes. Also confirmed training curves are converged (val error plateaus well before the 1800-epoch schedule ends) — this isn't a "needs more training" artifact.

**Recommended write-up framing:** report the tied-with-FNO checkpoint as the headline Burgers result (unchanged), and the ablation matrix as a separate mechanism/generalization section — not folded into the headline comparison.

## TODO

- [ ] Write the Assumption 2.2 (shock-exclusion) limitation sentence for Discussion/Limitations.
- [ ] Decide on citing Calvello et al. / Zhang-Leung-Schaeffer in Related Work.
- [ ] Finish Introduction paragraph 2 ("the gap") now that Related Work citations are locked.
- [ ] Once Darcy runs finish: run the resolution-transfer sweep, write up raw/normalized channel ablation, get parameter counts, note the LNO decoupled-transfer caveat (only `LNO_triple` supports decoupled encode/decode query positions).
- [ ] Write the Experiments section's Burgers subsection (headline tie + ablation matrix, two-part framing above).
- [ ] Update the NS2d write-up to the new AE-reconloss primary checkpoint (1.61x/1.93x).
- [ ] Optional/stretch: 2D resolution-transfer ablation for NS2d (not started, not required for the core story).
- [ ] Abstract — write last, once Experiments numbers are final.
- [ ] Figure 1 — deferred, not started.
- [ ] Reproducibility statement + AI-use statement — placeholders, not written.
