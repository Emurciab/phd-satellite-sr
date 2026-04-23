"""
Uncert_Quant — Uncertainty Quantification via Heteroscedastic Deep Ensembles
=============================================================================

Implements a proper uncertainty quantification framework for the
MultiOutput SchedPer Direct + LearnedUps super-resolution model.

Approach: Heteroscedastic Deep Ensemble (Lakshminarayanan et al., NeurIPS 2017)
    - M ensemble members (default 5), each with variance heads
    - Each member outputs per-pixel mean μ(x) AND log-variance log σ²(x)
    - Trained with Gaussian NLL loss so each member learns aleatoric noise
    - Epistemic uncertainty from inter-member disagreement

Decomposition:
    Aleatoric  = (1/M) Σ_m σ²_m(x)           (mean predicted variance)
    Epistemic  = (1/M) Σ_m (μ_m(x) - μ̄(x))²   (variance of means)
    Predictive = Aleatoric + Epistemic

Modules:
    train_het_ensemble_direct_learnedUps.py
        - HetMultiOutputTemporalSRDirectLearnedUps  (model with variance heads)
        - VarianceHead                               (log-variance prediction)
        - HeteroscedasticMultiOutputLoss             (Gaussian NLL + aux losses)
        - Training loop with --seed / --ensemble_id support

    evaluate_ensemble.py
        - Loads M trained members and produces uncertainty maps
        - Computes per-sample PSNR/SSIM/RMSE + aleatoric/epistemic/predictive
        - Error-uncertainty correlation analysis

    run_ensemble.sh
        - Convenience script to train all M members sequentially

IMPORTANT: This folder is self-contained. No files in the parent directory
were modified. Building blocks (EncoderBranch, ProgressiveUpscalerLearned,
etc.) are imported from the parent's satellite_sr_model_v2*.py files.
"""

from .train_het_ensemble_direct_learnedUps import (
    HetMultiOutputTemporalSRDirectLearnedUps,
    VarianceHead,
    HeteroscedasticMultiOutputLoss,
)
