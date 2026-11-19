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

# Semantic FoodMapper

This Shiny for Python app matches food descriptions across two CSV files. It embeds descriptions with GTE-Large
and uses cosine similarity to pick the best reference description for each input.

## Run with Docker

The Dockerfile uses Python 3.12. Build and start the app from this repository.

```sh
docker build -t semantic-food-mapper .
docker run --rm -p 127.0.0.1:7860:7860   -e MODEL_FALLBACK_MODE=local -e DEEPINFRA_API_KEY=unused semantic-food-mapper
```

Open http://127.0.0.1:7860. With the default asynchronous path, `MODEL_FALLBACK_MODE=local` runs the
Sentence Transformers model in the app and can download weights on first use. The current UI still checks for a
nonempty key before running, so the local example supplies a placeholder. Leave `EMBEDDING_ASYNC` at its default
for this mode.

To use DeepInfra, set `DEEPINFRA_API_KEY` in the host environment and pass it with `-e DEEPINFRA_API_KEY`.
The default `MODEL_FALLBACK_MODE=auto` sends the selected descriptions to DeepInfra for embeddings and uses
the local model if the API fails.

## Match two files

1. Upload an input CSV and a reference CSV with headers, or load the built-in sample.
2. Select the description column in each file. Preview optional text cleaning.
3. Run matching. The default threshold is 0.85. Inspect the suggested reference, score, and match status.
4. Export all results or the matched rows. When input cleaning is enabled, the all-results CSV uses the cleaned
   input text in the selected column.

Use the result charts to examine score distributions and match counts. This repository contains the web app.
The separate [FoodMapper desktop app](https://foodmapper.app) has a native macOS interface and additional review tools.

## License

[CC0 1.0](LICENSE).

Developed at the USDA Agricultural Research Service, Western Human Nutrition Research Center, Davis, California.
