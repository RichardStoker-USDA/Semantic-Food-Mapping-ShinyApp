---
title: Semantic FoodMapper
emoji: 🖇️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
header: mini
models:
  - thenlper/gte-large
tags:
  - food-science
  - food-database
  - semantic-search
  - usda
  - nutrition
  - match-food-databases
  - python
  - shiny
license: cc0-1.0
---

# Semantic Food Mapping ShinyApp

A web-based tool for matching food descriptions across nutritional databases using neural embeddings. This is the lightweight web version of FoodMapper, built with Shiny for Python.

[![License: CC0](https://img.shields.io/badge/License-CC0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Shiny for Python](https://img.shields.io/badge/Shiny-for%20Python-blue)](https://shiny.posit.co/py/)

## Looking for the Full App?

The full-featured **FoodMapper** desktop application for macOS is available at **[foodmapper.app](https://foodmapper.app)**. It includes match review workflows, local MLX-powered processing, and a native macOS interface. If you need more than just embedding and matching, that's where to go.

## What This Web Version Does

This app handles the core embedding and semantic matching pipeline: upload two CSV files, pick your text columns, and the tool matches items across databases using GTE-Large embeddings and cosine similarity. It won't do match review or editing like the desktop app, but it's useful for quick matching jobs and exploratory work.

### [Try it online on HuggingFace Spaces](https://huggingface.co/spaces/richtext/semantic-food-mapper)

No installation needed. The hosted version is ready to go.

## Research Context

Developed at the USDA Agricultural Research Service as part of research on automated dietary data mapping.

**Research Paper**: [Title Placeholder - Publication Pending]
**Authors**: Lemay DG, Strohmeier MP, Stoker RB, Larke JA, Wilson SMG

## Features

- Semantic matching with GTE-Large neural embeddings
- Batch processing for thousands of items
- 7 interactive chart types for exploring match distributions
- Optional text cleaning with live preview
- CSV export with all original data preserved
- API processing with automatic CPU fallback

## Running Locally

### Prerequisites

- Python 3.8+
- 2GB RAM minimum (4GB recommended for CPU mode)

### Setup

1. Clone and install:
```bash
git clone https://github.com/RichardStoker-USDA/Semantic-Food-Mapping-ShinyApp.git
cd Semantic-Food-Mapping-ShinyApp
pip install -r requirements.txt
```

2. Install PyTorch for your platform:

**Mac (Apple Silicon):**
```bash
pip install torch torchvision torchaudio
```

**NVIDIA GPU (CUDA):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

3. Run:
```bash
export MODEL_FALLBACK_MODE="local"  # Force local GPU/CPU usage
shiny run app.py --port 8000
```

4. Open `http://localhost:8000`

### Processing Modes

**HuggingFace Spaces**: Uses API for fast processing with CPU fallback. No setup.

**Local GPU**: With `MODEL_FALLBACK_MODE="local"`, the app uses Metal (Apple Silicon), CUDA (NVIDIA), or CPU as fallback.

## Usage

1. Upload two CSV files (input items and target database)
2. Select the text columns to match on
3. Set your similarity threshold (default: 0.85)
4. Run matching and review results
5. Download matched results as CSV

Sample data is included if you want to test without your own files.

## Technical Details

- **Framework**: Shiny for Python
- **Embedding Model**: [thenlper/gte-large](https://huggingface.co/thenlper/gte-large)
- **Visualizations**: Plotly
- **Data Processing**: Pandas, NumPy

## Deployment

The app is hosted on HuggingFace Spaces with automated deployment via GitHub Actions. Push to GitHub and the Space rebuilds automatically.

## Development Team

**Principal Investigator**: Dr. Danielle G. Lemay
**Developer**: Richard Stoker
**Organization**: USDA Agricultural Research Service
**Location**: Western Human Nutrition Research Center, Davis, CA

## Citation

If you use FoodMapper in your research, please cite:

[Citation placeholder - to be updated upon publication]

## License

This work is in the public domain in the United States because it is a work prepared by an officer or employee of the United States Government as part of that person's official duties. See [17 U.S.C. 105](https://www.copyright.gov/title17/92chap1.html#105).

Internationally, this work is released under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).

## Contact

Richard Stoker
IT Specialist (Scientific)
richard.stoker@usda.gov
USDA ARS Western Human Nutrition Research Center

## Acknowledgments

This research was supported by USDA ARS project funding. The application uses the GTE-Large model developed by Alibaba DAMO Academy.
