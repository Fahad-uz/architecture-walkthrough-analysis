# Local segmentation candidates

Reviewed15October2026. This is source/runtime/availability assessment,not a
measured model comparison. No weights or dependencies were downloaded and no
model was adopted. Published model scores below are not PlanStride results.

## Existing provider status

The current application uses local wall-band vision. CubiCasaGeometryProvider
has inference_implemented=False; providing segmentation_checkpoint does not
activate model inference. setup_cubicasa.py now reports this before optional
imports or downloads, rather than implying that a checkpoint enables the model.
The ten setup/provider/sanity tests pass,including no-download behaviour.

## CubiCasa5K author baseline

The [official repository](https://github.com/CubiCasa/CubiCasa5k) supplies an
hourglass multitask model and a linked checkpoint. Its documented environment
uses Python3.6.5,PyTorch1.0 and CUDA; evaluation calls model.cuda(). A modern CPU
or MPS experiment therefore needs a reviewed compatibility adapter,not the
unmodified evaluation command. The repository's [license](https://raw.githubusercontent.com/CubiCasa/CubiCasa5k/master/LICENSE)
is CC BY-NC4.0. Keep any evaluation assets restricted/noncommercial and outside
distributable project files. Establish training-split overlap before interpreting
our CubiCasa development agreement as generalization. Size and local runtime
cost of this checkpoint have not been measured.

## OpenBIM FloorPlan AI

The author's [README](https://github.com/Chunling1/OpenBIM-FloorPlan-AI) describes
four-class background/wall/window/door segmentation and an ONNX deployment path.
[Releasev1.0.0](https://github.com/Chunling1/OpenBIM-FloorPlan-AI/releases/tag/v1.0.0)
actually exposes three PyTorch checkpoints:31,117,194bytes,97,923,760bytes and
75,275,273bytes. No ONNX asset was listed in that release when checked,so our
installed ONNX runtime alone cannot run those published files. The code has an
[MIT license](https://github.com/Chunling1/OpenBIM-FloorPlan-AI/blob/main/LICENSE),
while the README reports training on CubiCasa5K; explicit checkpoint-use terms
and training overlap remain to be established. Publisher speed/accuracy claims
have not been reproduced here. Do not treat them as a measured benefit.

## WAFFLE

The authors' [official repository](https://github.com/TAU-VAILab/WAFFLE) links
finetuned DETR,CLIPSeg and wall-detection checkpoints and a building-diverse
floor-plan dataset with Wikimedia source URLs and building identifiers. Its
README points to Wikimedia licensing rather than a single explicit checkpoint
license. Check per-image/model terms and overlap before acquiring or integrating
assets. The diffusion-based generation paths must not invent structural geometry
for this project. No local inference or resource-cost measurement was performed.

## Next controlled model experiment

Verify a checkpoint's exact source,license,hash and trained-source overlap,then
run a small isolated development-only comparison before installation in the app.
Record preprocessing,output classes,pixel alignment,runtime,memory,failures and
wall/opening agreement against the unchanged baseline. Keep all30quarantined
plans untouched. Do not adopt a model without demonstrated benefit and clear
use terms. Do not run the current setup script as an inference solution.
