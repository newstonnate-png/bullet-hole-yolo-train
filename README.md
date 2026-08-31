# bullet-hole-yolo-train

Trains a single-class bullet-hole detector on a rented Vast.ai GPU. The job downloads the datasets, runs two training passes, exports ONNX, and pushes the weights to a private Hugging Face repo. It trains the model only. The application that scores shots lives elsewhere.

There is no image to build. The job uses the stock `ultralytics/ultralytics:latest` image with `scripts/provision.sh` on top.

## Why two phases

The detector answers one question: hole or no hole. No zones, no colours, no score. Those come later, in whatever application consumes the model. It samples the target's colour around each hole to decide the zone. Dataset choice here only affects hole localization.

No public dataset matches every rig: the lens, the crop, the lighting, the lane distance. So training runs in two steps.

1. **Phase 1, pretrain.** `datasets/pool/` holds public bullet-hole sets (CC BY 4.0 works well), merged and reduced to one class. This teaches the model what a hole looks like in general. Where to find them and how to name them is in `templates/bullet-hole-yolo-train.md`.
2. **Phase 2, fine-tune.** `datasets/finetune/` holds 200 to 400 frames from the camera you will actually use, shot through the same lens at the same printed target. Save them cropped and rotated exactly as the scoring application sees them, then label them in Roboflow. Fewer epochs, lower learning rate, frozen backbone. If there is no fine-tune set, the script skips phase 2 and you get a baseline model.

## What you need

- Vast.ai account and the `vastai` CLI.
- Roboflow API key. Exporting a dataset is free; hosted inference and Roboflow Train are the paid parts and this repo uses neither.
- Hugging Face token with write access, plus the name of a private model repo for the artifacts.

## Run a training job

1. Copy `.env.example` to `.env` and fill in your own values. The `.env` file is gitignored. Real tokens do not go in any tracked file.
2. Open `templates/bullet-hole-yolo-train.md`, copy the Docker options line, and swap each `REPLACE_ME` and `your-...` value for the matching value from your `.env`. Paste the finished line into the Vast console. Do not save filled values back into the template.
3. Create the instance: image `ultralytics/ultralytics:latest`, launch mode **docker ENTRYPOINT**, 40 GB disk, one RTX 3090 or 5090. Add the on-start script and the line from step 2.
4. `scripts/provision.sh` runs the rest: dependencies, `fetch_dataset.py`, `train.py` (both phases plus ONNX export), `upload_artifacts.py`, then JupyterLab on port 8888.
5. When `results.png` has settled and `best.onnx` is in the HF repo, destroy the instance with `vastai destroy`.
6. To improve the model, label real frames, publish them as a new Roboflow version, then rerun with `-e FINETUNE_SOURCE=roboflow:<ws>/<proj>/<ver>`.

## Config

`configs/train_config.yaml` holds the defaults. Any environment variable with the upper-cased key overrides its entry: `MODEL`, `IMGSZ`, `EPOCHS`, `FINETUNE_EPOCHS`, `BATCH`, `PATIENCE`, `DEVICE`, `CACHE`, `RUN_NAME`. `PRETRAIN_SOURCES` and `FINETUNE_SOURCE` choose the datasets. See `docs/MODEL_NOTES.md` for model and resolution choices, and `docs/ADDING_CAMERA_FRAMES.md` for the capture-and-label loop.

## Handing the model to a scoring app

The exported `best.onnx` is a detector, not a scorer. Anything that consumes it needs four things:

- **One class.** Every surviving box is `bullet-hole`. Zones and points belong to the consumer, usually by sampling the target colour in a ring around the hole centre.
- **A fixed input size.** The ONNX bakes in `IMGSZ` (960 by default). Train and infer at the same size.
- **Raw output.** The tensor is `[cx, cy, w, h, conf]` per anchor. NMS is not part of the export; the consumer applies it.
- **A CPU budget.** With a static target and frame skipping in the consumer, 2–4 effective FPS is usually enough. The timings in `docs/MODEL_NOTES.md` are a starting point, not a promise.

Capture fine-tune frames the way the consumer will see them: same crop, rotation and zoom. Otherwise the model learns a view it never gets at scoring time.

## Local dry run, no GPU

```
pip install ultralytics onnxruntime pyyaml
# put a handful of images+labels in datasets/pool/train/{images,labels}
python scripts/make_val_split.py --base datasets/pool
MODEL=yolo11n.pt EPOCHS=1 IMGSZ=320 python scripts/train.py
# optional: add datasets/finetune/train/{images,labels} + a data/finetune.yaml
#           (copy data/pool.yaml, repoint `path`) to exercise phase 2
```

This checks the wrapper, config parsing, and ONNX export before you spend money on GPU time.

## Layout

```
configs/train_config.yaml    defaults, overridable by env vars
data/pool.yaml               phase-1 data config (written by fetch_dataset.py)
data/finetune.yaml           phase-2 data config (written when FINETUNE_SOURCE is set)
data/bullet-hole.yaml        legacy alias for pool.yaml (placeholder in git)
scripts/provision.sh         Vast on-start entrypoint
scripts/sources.py           source-spec parsing + download/normalize/merge helpers
scripts/fetch_dataset.py     build datasets/pool/ and datasets/finetune/
scripts/make_val_split.py    deterministic 85/15 split for a --base dir
scripts/train.py             two-phase ultralytics train + ONNX export
scripts/upload_artifacts.py  push weights/onnx/plots to a private HF model repo
templates/                   raw Vast launch config (hand-edited, source of truth)
docs/                        model notes + dataset loop
.env.example                 every env var the job reads, with placeholder values
```
