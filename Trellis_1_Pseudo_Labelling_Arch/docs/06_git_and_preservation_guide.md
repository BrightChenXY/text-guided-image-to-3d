# 06. Git / Preservation / Reviewer-Ready Repository Guide

This file is adapted to the actual Architecture 1 work, while preserving the same preservation philosophy used in the earlier Git push notes.

## 1. What should go into the Git repository

The Git repository should contain as much of the *reviewable* project state as possible, including:

- training configs,
- modified dataset/model/trainer files,
- documentation,
- plotted figures,
- smaller logs,
- lists/manifests of the workspace,
- helper scripts for reproducing graphs,
- and, if size allows, smaller checkpoints or selected checkpoints through Git LFS.

## 2. What should not be relied on in Git alone

GitHub is not enough to preserve the *entire* research workspace, especially if it contains:

- many large `.pt` checkpoint files,
- large latent datasets,
- raw generated 3D assets,
- large caches,
- model downloads,
- or full workspace backups.

So preservation should happen in two layers:

### Layer A — GitHub repo

For code, docs, plots, important small artifacts, and selected checkpoints.

### Layer B — Workspace archive backup

For the full project state, including anything too large or too messy for GitHub.

## 3. Reviewer-friendly repo structure

A reviewer-friendly exported repo should ideally include folders like:

- `README.md`
- `docs/`
- `docs/figures/`
- `artifacts/`
- `scripts/`
- `configs/`
- relevant modified `trellis/` files or a clearly documented patch list

## 4. What to say about the repo contents

The repo should clearly explain that it contains:

- the Architecture 1 method description,
- the run/debugging chronology,
- the loss-curve generation script,
- and enough notes that a reviewer can understand *both* the research idea and the engineering steps taken.

## 5. Backup strategy that should be preserved

The preservation note in the earlier Git push document recommended two backup layers:

1. make a backup archive of the reviewer-ready repo,
2. make another backup archive of the larger workspace.

That is still the correct strategy for this project.

## 6. What to include in the repo README

The README should make the following explicit:

- this is TRELLIS 1 Architecture 1 work,
- it uses pseudo-3D labels from frozen TRELLIS inference,
- the proof-of-concept runs were small-scale and debugging-heavy,
- the strongest preserved result is the successful Stage 2 500-step run and its loss curve,
- the repository is meant to be understandable to someone who did not participate in the debugging.

## 7. What to preserve outside GitHub

If the raw workspace still exists, preserve separately:

- the `TRELLIS/` codebase with patches,
- run directories under `arch1_runs/`,
- the filtered/preprocessed subset,
- any generated pseudo-3D assets,
- any checkpoint directories,
- and any manually generated plots.

## 8. Reviewer summary

The point of the repository is not only to store code. It is to preserve:

- what was attempted,
- what actually worked,
- what failed and why,
- and what the next person should do from here.

