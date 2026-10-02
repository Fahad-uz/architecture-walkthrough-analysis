# Real-plan research data

This collection supports local, noncommercial development of PlanStride. It is
not a claim that arbitrary uploaded plans reconstruct accurately. Dataset images,
annotations and derived models stay under ignored `outputs/benchmark-data/` and
must not be bundled with the application or pushed to GitHub.

## CubiCasa5K: selected source

- Official project: <https://github.com/CubiCasa/CubiCasa5k>
- Version 1.0 dataset and citation: <https://zenodo.org/records/2613548>
- License: <https://github.com/CubiCasa/CubiCasa5k/blob/master/LICENSE>
- Paper: <https://arxiv.org/abs/1904.01920>

The project specifies CC BY-NC 4.0. Keep the attribution and license with local
copies; commercial use needs a separately appropriate permission or dataset.
The source contains images and polygon annotations in SVG. Its three drawing
styles provide useful variety, but its paper describes predominantly Finnish
real-estate material. They do not represent all countries, languages, photographs,
hand sketches, elevations or measurement conventions.

The default acquisition selects 70 official training examples for development
and 30 official test examples for quarantine, balanced across `colorful`,
`high_quality` and `high_quality_architectural`. It downloads only `F1_scaled.png`
and `model.svg` per chosen directory. The upstream archive is 5,469,495,706 bytes;
HTTP byte ranges avoid downloading it in full. Each member is checked against
its upstream ZIP CRC and size, then gets a SHA-256 record. The archive's published
MD5 is recorded for provenance but is **not** reported as verified by a partial
download.

From the repository root, with existing project dependencies installed:

```sh
.venv/bin/python tools/benchmark/acquire_cubicasa.py
```

The reproducible manifest is
`outputs/benchmark-data/cubicasa5k/manifest.json`. It lists every downloaded image,
SVG, official split list and license, with source, bytes and checksums. The
acquisition writes progress after each example and verifies cached files when
resumed. The default selection is deterministic; do not change it after viewing
evaluation results. No model weights or additional software are required.

## Split and annotation limitations

Quarantined examples are **not yet a certified final evaluation set**. The paper
describes random official splits, not building-disjoint or publisher-disjoint
splits. The archive does not provide a verified building or source-family mapping
in its split lists. Numeric directory IDs are kept disjoint, with one floor image
per directory, but those IDs cannot establish independent buildings. The three
style folders are sampling strata, not independent source families.

The manifest therefore explicitly leaves `building_id` and `source_family`
unknown. It flags exact pixel duplicates and possible near duplicates using a
difference hash. That screen is incomplete evidence: visual family review and
additional provenance are still required, including possible rotated, cropped
or redrawn versions. Do not tune the detector, inspect its outputs, or report a
final accuracy number on the quarantine set until those issues are resolved.

The supplied SVG annotations have not been converted into our evaluation schema
or independently checked here. Evaluate development examples first, establish
coordinate alignment and metrics, and review ground truth separately from
predictions. Never interpret a successful pipeline run as accurate walls, rooms,
openings, scale or a correct walkthrough.

## FloorPlanCAD: download deferred

Official source: <https://floorplancad.github.io/>

The site lists SVG/PNG drawings and train/test downloads, but explicitly limits
its CC BY-NC 4.0 license to annotations and the website and says its authors do
not own the drawings' copyright. It also reports that the project shut down in
early 2022. Consequently, no FloorPlanCAD assets were downloaded. Obtain clearer
rights for the drawings before importing them into this collection; the
annotation license alone does not establish those rights.

## Work still required

- Independently establish building/source-family groups and review duplicate
  candidates before assigning a final, untouched evaluation split.
- Add permitted examples from other sources and capture domains; keep all
  transformations and redraws of one plan in its original group.
- Validate wall, room, opening and scale annotations in image coordinates.
- Report coverage, failures and correction effort alongside geometric metrics;
  preserve the private user plan as a separate regression case.
