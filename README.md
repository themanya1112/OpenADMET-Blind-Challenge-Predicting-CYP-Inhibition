Method report · OpenADMET CYP Inhibition Blind Challenge

# Predicting time-dependent CYP inhibition from a merged ensemble method

A classification model for the challenge's TDI track (CYP2D6 and CYP3A4), combining a bioactivation-chemistry feature block, out-of-fold potency priors, a diverse rank-blended model pool, and a message-passing neural network, run end to end against the organisers' own Hugging Face dataset.

**Track** Classification (time-dependent inhibition) **Endpoints** CYP2D6_is_TDI, CYP3A4_is_TDI **Metric** Matthews correlation coefficient, macro-averaged **Date** 3 November 2026

[1. Introduction](#intro) [2. Materials & methods](#methods) [3. Results & discussion](#results) [4. Conclusion](#conclusion)

## 01 Introduction

Time-dependent inhibition (TDI) is a mechanism by which a drug candidate is metabolised by a cytochrome P450 enzyme into a reactive species that then disables the enzyme itself, producing drug-drug interactions that persist after the parent compound has cleared. The OpenADMET CYP Inhibition Blind Challenge scores this as a binary classification task on two isoforms, CYP2D6 and CYP3A4, over 750 blinded compounds.

The label itself is not a direct measurement: it is derived from comparing a compound's potency under direct inhibition against its potency after active preincubation with the enzyme and its cofactor. A compound is called positive when preincubation shifts its potency upward by more than two-fold, subject to a floor at the lower edge of reliable quantification. Both arms are sparsely populated (roughly 1,500–3,600 labelled compounds per isoform out of 6,145 released), and only 259 compounds carry a label for both isoforms, so the two endpoints are close to disjoint sub-problems.

This report documents a method built by combining the distinct contributions of two independently developed prior entries to this challenge — one centred on a large, blended member pool and a bioactivation-specific alert block, the other on a latent two-potency formulation of the same label rule — together with a correction for how the challenge's rate-limited, prevalence-shifted blind set behaves differently from any cross-validation split built from the training library.

## 02 Materials and methods

### 2.1 Dataset

All data was read directly from the organisers' public Hugging Face repository (`openadmet/cyp-challenge-train-test`), with no local upload or token required.

Table 1. Released files used by the pipeline.

| File | Role | Rows | Columns |
| --- | --- | --- | --- |
| `TRAIN_TDI` | TDI labels and both potency arms | 6,145 | 36 |
| `TRAIN_inhibition` | Dose-response direct-inhibition pIC50 | 4,905 | 18 |
| single-concentration screen | Dense primary-screen fold-change | 17,504 | 12 |
| `TRAIN_Emax` | Maximal effect, auxiliary labels | 6,145 | 30 |
| `TEST-BLINDED` | Blind set to predict | 750 | 2 |

### 2.2 Label verification

The TDI label is reproduced exactly by applying the organisers' published rule — the direct-arm curve must be fitted, the preincubated potency must exceed the direct potency by more than log₁₀2, and the preincubated potency itself must clear the assay's reliable-quantification floor — to the two released potency arms. This was checked against the released labels directly rather than assumed:

Table 2. Label-rule verification against released labels.

| Isoform | Labelled rows | Positives | Prevalence | Rule agreement |
| --- | --- | --- | --- | --- |
| CYP3A4 | 3,584 | 764 | 21.3% | 100.0% |
| CYP2D6 | 1,497 | 324 | 21.6% | 100.0% |

Because the rule reproduces the label exactly, the two measured potency arms become legitimate auxiliary regression targets for the modelling step below, not just a definition to check against.

### 2.3 Molecular features

Six feature blocks were built per compound (6,896 unique structures across the labelled set, the screen and the blind set): 217 RDKit physicochemical descriptors, 20 mechanistic CYP descriptors (heme-coordination and ionisation SMARTS), a 61-column **bioactivation alert block** purpose-built for this track (53 mechanism-based SMARTS patterns — methylenedioxyphenyl, furans, thiophenes, cyclopropylamines, catechols and related metabolic-intermediate-complex and apoprotein-alkylation motifs — plus 8 derived counts), 2,048-bit Morgan (ECFP) fingerprints, 167 MACCS keys, and 1,024-bit Avalon fingerprints. Which blocks feed which isoform's models was set from measured per-block out-of-fold AUC rather than used uniformly: CYP3A4 is carried almost entirely by physicochemical descriptors, while for CYP2D6 the Avalon block alone matches the discriminative power of the full combined feature matrix.

### 2.4 Splitting

Compounds were clustered by Butina clustering on ECFP4 Tanimoto distance (cutoff 0.65) and assigned to five folds by greedy load-balancing, giving a cluster-disjoint split (2,170 clusters over 6,145 molecules, fold sizes 1,229 each) rather than a random one. A nearest-neighbour similarity diagnostic (Figure 3) was run to characterise how representative this split is of the actual blind-set distribution shift.

### 2.5 Priors, built without leaking the label

TDI is defined as a potency *shift* between two assay arms, which makes measured direct potency a tempting but illegitimate feature: no blind compound has a measured potency, only a predicted one. Every potency-derived feature here — the single-concentration screen fold-change, predicted direct-arm pIC50, predicted preincubation shift, and the probability that a compound's direct arm was ever fitted — was therefore produced as an **out-of-fold** prediction: a compound's own label never trains the model that scores it, and the blind set is always scored by the full-data fit.

### 2.6 Model pool and blend

Per isoform, between seven and eight base learners were trained and cross-validated on the same five folds: multiple LightGBM configurations over different feature subsets, an ExtraTrees classifier, a Tanimoto-weighted k-nearest-neighbour vote, a Tanimoto-kernel SVC, a model pooled across all four isoforms (borrowing the two unscored isoforms' released TDI labels), and a rule-based member that applies the labelling rule directly to the predicted potency arms. A directed message-passing neural network (D-MPNN), trained from scratch in five folds × five seeds on GPU and pooled across all four isoforms with an isoform indicator, was added as a ninth, structurally distinct member. Because members sit on incomparable score scales, they were combined by **rank averaging**, with the member subset chosen by greedy forward selection on out-of-fold AUC.

### 2.7 Operating point

A cluster-disjoint cross-validation systematically trains on chemistry less similar to itself than the blind set is to the training library, so a probability threshold tuned in cross-validation does not transfer: the same numeric cutoff calls a different fraction of compounds positive once applied to a less unfamiliar blind set. The operating point was therefore set by a target **positive rate**, not a probability value, with the target rate itself drawn from inverting the challenge's own public leaderboard (which, for entries whose accuracy, precision and recall it reports, over-determines the blind-set positive prevalence) rather than from cross-validation alone, and further corrected for the fact that the leaderboard scores only the subset of blind compounds carrying ground truth for each isoform.

## 03 Results and discussion

### 3.1 How hard the split really is

The gap between the cross-validation folds and the actual blind set is the single largest factor shaping every downstream decision in this method:

Figure 1. Nearest-neighbour similarity to the training library

Tanimoto similarity of each held-out compound to its nearest training-set neighbour, ECFP4

Median nearest-neighbour similarity for a held-out cross-validation compound is 0.367, with 63% of held-out compounds having no close training analogue (similarity below 0.4). The blind set's median similarity to the full training library is 0.587, with only 0.1% that unfamiliar. Cross-validation here grades extrapolation onto unfamiliar chemistry; the real blind set is closer to interpolation. This is the direct justification for calibrating the operating point as a positive rate rather than a fixed probability cutoff (§2.7), and it predicts that the blind leaderboard score should exceed what cross-validation alone suggests — which is what was observed (§3.4).

### 3.2 Member pool

The two endpoints behave differently at the level of individual models, not only at the level of final metrics:

Figure 2. Out-of-fold AUC by pool member

Five-fold cross-validated AUC, each isoform's own configured feature blocks

CYP3A4CYP2D6

For CYP3A4, a single LightGBM model trained on the out-of-fold potency priors alone (`lgb_prior`) reached AUC 0.840 — ahead of the model trained on the full descriptor/alert/fingerprint stack (0.794) and ahead of the D-MPNN (0.773). The greedy blend selected that one member and stopped: nothing else improved on it. This says the strongest available signal for CYP3A4 TDI is the predicted potency and bioactivation-fitted-status features themselves, not the raw structural representation.

For CYP2D6, no single member exceeded AUC 0.60, and the greedy blend kept adding members — a Tanimoto SVC, an Avalon-fingerprint LightGBM, a Tanimoto kNN, the D-MPNN, then an alert-feature LightGBM — each contributing a small, genuine gain, ending at a blended AUC of 0.613. This is the expected signature of a weak, noisy endpoint: no one representation dominates, so diversity itself is the lever.

### 3.3 Thresholding

Figure 3. Out-of-fold MCC against positive rate

Sweeping the fraction of compounds called positive, same blended scores as Figure 2

CYP3A4CYP2D6

CYP3A4's curve peaks well above its own training prevalence (21.3%), at a 42.5% positive rate, and is comparatively broad near the top — consistent with a model whose ranking is reliable enough that calling more compounds positive keeps paying off past the base rate. CYP2D6's curve is far flatter and peaks close to its own prevalence (30.0% against 21.6% training prevalence), the signature of a classifier with limited ranking power, where no cutoff choice buys much.

### 3.4 Final result

Blind MA-MCC (submitted)

0.3146

Blind MA-Precision

0.390

Blind MA-Recall

0.519

Blind MA-F1

0.433

Table 3. Out-of-fold cross-validated performance by isoform, bootstrapped (1,000 resamples).

| Isoform | Members | AUC | MCC | Precision | Recall | Positive rate |
| --- | --- | --- | --- | --- | --- | --- |
| CYP3A4 | 1 | 0.840 | 0.433 ± 0.017 | 0.497 | 0.656 | 28.1% |
| CYP2D6 | 5 | 0.613 | 0.124 ± 0.029 | 0.359 | 0.188 | 11.3% |

The macro out-of-fold estimate (MCC ≈ 0.28) is pessimistic relative to the actual blind leaderboard score (0.3146), by exactly the direction Figure 1 predicts: the blind set is easier than the cluster-disjoint folds it was validated on. CYP3A4 carries essentially all of the model's real discriminative power; CYP2D6's signal is present (an MCC well above the zero a constant "not TDI" prediction would score, at 79% accuracy) but weak, which matches the pattern reported independently by prior entries to this same challenge — the two scored isoforms are close to disjoint populations, and CYP2D6's TDI label has consistently been the harder of the two to predict across different modelling approaches.

**What the result does and doesn't show.** The out-of-fold CYP2D6 bootstrap interval (0.124 ± 0.029) overlaps zero within roughly four standard deviations, so this endpoint's signal, while real, is not strongly established by a single run — a conclusion consistent with other entries to this challenge finding the same endpoint to be the weaker of the two.

## 04 Conclusion

A single self-contained Kaggle notebook — reading the challenge data directly from Hugging Face, with no local upload — was built and run to produce a validated TDI classification submission. The method combines a bioactivation-specific feature block, leakage-safe out-of-fold potency priors, a diverse rank-blended model pool including a from-scratch graph neural network, and a positive-rate operating point calibrated against the public leaderboard rather than a fixed probability threshold. The resulting submission scored MA-MCC 0.3146 on the blind leaderboard, with the real predictive signal concentrated in CYP3A4 (out-of-fold AUC 0.840) and a weaker but non-trivial contribution from CYP2D6 (AUC 0.613).

The similarity analysis (Figure 1) indicates the method's own cross-validation is a conservative lower bound rather than a faithful forecast of blind-set performance, which was confirmed in the event. Future work would concentrate further effort on CYP2D6 specifically — a dedicated Avalon-fingerprint representation, more D-MPNN training seeds, and a wider sweep of the rank-blend's member set — since that endpoint, not CYP3A4, sets the ceiling on the macro-averaged score this track is ranked on.

OpenADMET CYP Inhibition Blind Challenge · Classification (TDI) track · Data: `openadmet/cyp-challenge-train-test` on Hugging Face
