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

# State of the Field

Data assimilation has a long history in the geosciences. Operational systems such as
ECMWF's IFS use 4D-Var [@rabier2000ecmwf], while ensemble Kalman filter (EnKF)
methods are widely used in both atmospheric and ocean modeling [@evensen2003ensemble].
Open-source DA frameworks like DART [@anderson2009dart] and JEDI [@jedi2023] provide
flexible toolkits for traditional DA, but they are built around dynamical model
integration and do not support learned priors.

The intersection of deep learning and DA is an active research frontier. Neural
network--based approaches range from learning surrogate observation operators
[@arcucci2021deep] to fully replacing the background model with a generative prior.
Score-based diffusion models [@song2021scorebased; @ho2020denoising] are particularly
well suited for this role because they learn the full data distribution rather than a
point estimate, enabling probabilistic posterior sampling. @manshausen2024generative
demonstrated that a diffusion model trained on climate reanalysis fields can
assimilate sparse weather station observations with quality comparable to traditional
methods while providing calibrated uncertainty quantification.

Despite this promise, no existing Python library provides a turnkey pipeline for
diffusion-based DA that is accessible to climate scientists without deep ML expertise.
`DeepAssimilate` fills this gap by combining architecture search, diffusion training,
and score-based posterior sampling into a single, modular framework built on the
widely adopted Hugging Face Diffusers ecosystem.

# Software Design

`DeepAssimilate` is organized into three decoupled modules corresponding to the
workflow steps, each exposing a single high-level function call:

- **`da.search_architecture`** wraps the neural architecture search. It adapts the
  autonomous experimentation paradigm of @karpathy2026autoresearch: each candidate
  UNet configuration is trained for a fixed wall-clock budget (default 5 minutes),
  evaluated on validation MSE, and kept or discarded based on improvement over the
  current best. Results are logged to a TSV file for reproducibility.
- **`da.train_unconditional`** handles diffusion model training. It accepts any model
  architecture and noise scheduler from the Diffusers library, with built-in presets
  for common configurations (DDPM, EDM). Training supports mixed precision,
  gradient accumulation, and checkpoint saving.
- **`da.run_data_assimilation`** performs score-based posterior sampling. At each
  reverse-diffusion step $t$, the unconditional sample $x_t$ is corrected using the
  gradient of the observation log-likelihood:

$$x_t \leftarrow x_t + \sigma(t) \cdot \nabla_{x_t} \log p(y \mid x_t)$$

where $p(y \mid x_t) = \mathcal{N}(y; H(x_t), \sigma_{\text{obs}}^2 + \gamma \cdot
(\sigma(t)/\mu(t))^2)$. The variance term combines observation noise
$\sigma_{\text{obs}}$ with a time-dependent regularization controlled by $\gamma$,
ensuring stable conditioning at high noise levels. The scheduler type (EDM or DDPM)
is auto-detected, so users need not manage noise schedule internals.

Observation operators are implemented as callable objects with a common interface.
Built-in operators include masked gridded observations (random or station-based) and
linear operators, with an extensible base class for custom forward models. NaN
values in observation tensors automatically generate the corresponding mask.

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

# Research Impact Statement

`DeepAssimilate` is designed to accelerate research at the intersection of generative
modeling and geoscientific data assimilation. By lowering the implementation barrier,
it enables climate and weather researchers to experiment with diffusion-based DA
without building custom ML pipelines from scratch. Potential applications include
sparse observation networks (e.g., in data-scarce regions or for historical
reanalyses), rapid ensemble generation for uncertainty quantification, and benchmarking
learned priors against traditional DA methods. The modular design also makes it
suitable for educational use in graduate courses on data assimilation or scientific
machine learning.

# AI Usage Disclosure

Claude (Anthropic) was used to assist with code development, test writing, and
drafting of this manuscript. All AI-generated content was reviewed, verified, and
edited by the authors. The scientific methodology, experimental design, and
interpretation of results are entirely the work of the authors.

# Acknowledgements

We thank the developers of Hugging Face Diffusers and PyTorch for providing the
foundational tools on which `DeepAssimilate` is built. NCEP reanalysis data was
provided by NOAA PSL, Boulder, Colorado, USA.

# References
