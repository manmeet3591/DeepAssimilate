---
title: 'DeepAssimilate: A Python Framework for Diffusion-Based Generative Data Assimilation'
tags:
  - Python
  - data assimilation
  - diffusion models
  - weather
  - climate
  - machine learning
authors:
  - name: Manmeet Singh
    orcid: 0000-0002-3374-7149
    corresponding: true
    affiliation: 1
  - name: Alexandros Matakos
    affiliation: 2
  - name: Naveen Sudharsan
    affiliation: 3
  - name: Nico Renaldo
    affiliation: 2
  - name: Jaakko Santala
    affiliation: 2
  - name: Kim Kaisti
    affiliation: 2
affiliations:
  - name: Earth, Environmental and Atmospheric Sciences, Western Kentucky University, Bowling Green, KY, USA
    index: 1
  - name: Skyfora, Finland
    index: 2
  - name: Jackson School of Geosciences, The University of Texas at Austin, Austin, TX, USA
    index: 3
date: 22 March 2026
bibliography: paper.bib
---

# Summary

Data assimilation (DA) is a fundamental technique in weather forecasting and climate
science, combining observational data with numerical models to produce optimal estimates
of the state of the atmosphere and ocean. Traditional DA methods such as ensemble Kalman
filters and variational approaches require explicit knowledge of the model dynamics and
observation operators. Recent advances in score-based diffusion models have opened a new
paradigm: generative data assimilation, where a diffusion model trained on historical
data serves as the prior, and sparse observations are incorporated during the reverse
diffusion process via likelihood gradients [@manshausen2024generative].

`DeepAssimilate` is an open-source Python library that provides a complete, modular
pipeline for diffusion-based generative data assimilation. Built on PyTorch and Hugging
Face Diffusers, it offers a three-step workflow (\autoref{fig:workflow}):

1. **Architecture search**: An autoresearch-style [@karpathy2026autoresearch] neural
   architecture search that evaluates diffusion model architectures under a fixed
   training time budget, automatically selecting the best-performing configuration.
2. **Unconditional diffusion training**: Training of the selected diffusion model on
   gridded climate/weather data using any scheduler and architecture from the Diffusers
   library.
3. **Score-based data assimilation**: Zero-shot posterior sampling that combines the
   trained diffusion prior with sparse observations using the score function of the
   observation likelihood, following the approach of @manshausen2024generative and the
   NVIDIA PhysicsNeMo ReGen implementation.

![The three-step DeepAssimilate workflow. Step 1 performs neural architecture search
to identify the best diffusion model configuration. Step 2 trains the selected model
on gridded climate data. Step 3 assimilates sparse observations into the trained
diffusion prior via score-based posterior sampling.\label{fig:workflow}](fig_workflow.png)

# Statement of Need

The climate and weather science community increasingly recognizes the potential of
deep learning for data assimilation. However, existing implementations are often
tightly coupled to specific model architectures, require significant machine learning
expertise to configure, and lack the modularity needed for experimentation across
different datasets and observation configurations.

`DeepAssimilate` addresses these gaps by providing:

- **Accessibility**: Each of the three steps (search, train, assimilate) is designed as
  a one-liner function call, lowering the barrier for domain scientists.
- **Flexibility**: Any diffusion model architecture and noise scheduler from the
  Diffusers ecosystem can be used, from simple UNets to attention-augmented variants.
- **Reproducibility**: The architecture search produces a structured log of all
  experiments with keep/discard decisions, and the DA pipeline accepts random seeds for
  deterministic posterior sampling.
- **Observation handling**: Built-in support for sparse gridded observations (random
  masks, station locations), with an extensible observation operator interface for
  custom forward models.

# Key Features

The score-based data assimilation in `DeepAssimilate` implements the posterior
sampling approach from @manshausen2024generative. At each denoising step $t$,
the unconditional sample $x_t$ is corrected using the gradient of the observation
log-likelihood:

$$x_t \leftarrow x_t + \sigma(t) \cdot \nabla_{x_t} \log p(y \mid x_t)$$

where $p(y \mid x_t) = \mathcal{N}(y; H(x_t), \sigma_{\text{obs}}^2 + \gamma \cdot
(\sigma(t)/\mu(t))^2)$. The variance term combines observation noise
$\sigma_{\text{obs}}$ with a time-dependent regularization controlled by $\gamma$,
ensuring stable conditioning at high noise levels.

The architecture search module adapts the autonomous experimentation paradigm of
@karpathy2026autoresearch to the diffusion model setting: each candidate architecture
is trained for a fixed wall-clock budget (default 5 minutes), evaluated on validation
MSE, and kept or discarded based on improvement over the current best.

# Demonstration

We demonstrate `DeepAssimilate` using NCEP reanalysis 2-meter air temperature data
[@kalnay1996ncep] over North America on a 32$\times$32 grid. A UNet2D diffusion model
with 34.5M parameters is trained on daily snapshots from 2021--2024 using the
Heun discrete scheduler [@karras2022elucidating] with 1000 training timesteps.

\autoref{fig:da_results} shows the data assimilation results for a withheld test
sample from 2019. The analysis fields are generated by conditioning the diffusion
prior on randomly placed sparse observations at three densities (5%, 10%, and 50%).
Even with only 5% of grid points observed (approximately 50 stations over the domain),
the analysis achieves a spatial correlation of $r = 0.957$ with the truth. As
observation density increases, both the correlation improves and the RMSE decreases,
reaching $r = 0.988$ and RMSE = 4.4 K at 50% coverage.

![Score-based data assimilation results on NCEP 2-meter temperature. Top row: truth
and analysis fields at three observation densities. Bottom row: observation locations
(5% density) and error maps. Correlation ($r$) and RMSE are shown for each
case.\label{fig:da_results}](fig_da_results.png)

A key advantage of diffusion-based DA is the ability to generate multiple posterior
samples, providing uncertainty estimates. \autoref{fig:ensemble} shows an ensemble
of four posterior samples conditioned on 10% observations. The ensemble mean
achieves $r = 0.966$, while the ensemble spread (standard deviation across members)
reveals higher uncertainty in regions far from observations and lower uncertainty
near observed locations --- consistent with the expected behavior of a well-calibrated
probabilistic analysis.

![Ensemble data assimilation with four posterior samples conditioned on 10%
observations. (a) Truth field. (b) Sparse observations with station locations marked.
(c) Ensemble mean. (d) Ensemble spread showing uncertainty
estimates.\label{fig:ensemble}](fig_ensemble.png)

# Example Usage

```python
import deepassimilate as da

# Step 1: Find the best architecture
nas_result = da.search_architecture(
    train_dataset=train_ds, val_dataset=val_ds,
    cfg=da.NASConfig(time_budget_seconds=300, max_experiments=10)
)

# Step 2: Train the best model
model, scheduler, _ = da.train_unconditional(
    dataset=train_ds,
    cfg=da.UncondTrainConfig(architecture="edm_unet_2d", num_epochs=100)
)

# Step 3: Assimilate sparse observations
analysis = da.run_data_assimilation(
    model=model, scheduler=scheduler,
    observations=obs_with_nans,  # NaN = unobserved
    obs_noise_std=0.5, gamma=1e-3,
)
```

# Acknowledgements

We thank the developers of Hugging Face Diffusers and PyTorch for providing the
foundational tools on which `DeepAssimilate` is built. NCEP reanalysis data was
provided by NOAA PSL, Boulder, Colorado, USA.

# References
