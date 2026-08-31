# Bullet-Hole YOLO Training — Vast.ai Launch Template

This is a launch template for training a single-class bullet-hole detector on Vast.ai. Edit it only when the launch procedure changes. The values you paste into Vast live in the console, not in this file.

Every credential below is a placeholder. Real tokens stay in your local `.env` and in the Vast console. They never go in a commit.

## 1. What the job does

1. Clones this repository into `/workspace/t`.
2. Installs the training dependencies.
3. Downloads and merges the configured bullet-hole datasets.
4. Collapses the source annotations to one `bullet-hole` class.
5. Trains in up to two phases: a pretrain pass on the public pool, then an optional fine-tune pass on real camera frames.
6. Exports `best.onnx` for CPU inference.
7. Uploads weights, plots, metrics, and configuration to the private Hugging Face model repo when one is configured.

## 2. Repository map

| Path | Purpose |
|---|---|
| `configs/train_config.yaml` | Default model, resolution, epoch, batch, and run settings. |
| `data/bullet-hole.yaml` | Tracked placeholder for a dataset YAML. The fetch step generates the active YAML files. |
| `scripts/provision.sh` | Vast.ai entrypoint that runs the complete job. |
| `scripts/fetch_dataset.py` | Downloads, normalizes, merges, and splits datasets. |
| `scripts/train.py` | Runs the two training phases and exports ONNX. |
| `scripts/upload_artifacts.py` | Uploads the latest run artifacts to Hugging Face. |
| `docs/MODEL_NOTES.md` | Model and input-resolution guidance. |
| `docs/ADDING_CAMERA_FRAMES.md` | Workflow for adding real camera frames and fine-tuning. |
| `templates/bullet-hole-yolo-train.md` | This Vast.ai launch template. |
| `datasets/` | Downloaded training data; generated locally and ignored by Git. |
| `runs/` | Ultralytics weights, plots, metrics, and run configuration; generated locally and ignored by Git. |
| `.env.example` | List of every environment variable, with placeholder values. |

## 3. Before launching

Copy `.env.example` to `.env` and fill in your own values. Then paste those values into the Vast console. Leave this template with `REPLACE_ME` in place.

- `ROBOFLOW_API_KEY`
- `HF_TOKEN`
- `HF_MODEL_REPO`
- `JUPYTER_TOKEN`

The repository is private, so the Vast instance needs read access. Either make it public or use a clone URL with a read-only PAT:

```bash
git clone https://<user>:<PAT>@github.com/<owner>/bullet-hole-yolo-train ...
```

Keep that PAT out of the template too.

## 4. Vast.ai instance settings

- **Image:** `ultralytics/ultralytics:latest`
- **Launch mode:** docker ENTRYPOINT
- **Disk:** 40 GB
- **GPU:** RTX 3090 or 5090, 1x

### On-start script

Paste this into the Vast **On-start Script** field:

```bash
git clone https://github.com/<owner>/bullet-hole-yolo-train /workspace/t && bash /workspace/t/scripts/provision.sh
```

## 5. Dataset pool

`fetch_dataset.py` merges every spec in `PRETRAIN_SOURCES`, folds the classes into one `bullet-hole` class, and drops non-hole boxes such as target outlines and contours. You can mix any of these sources:

| Spec | What it means |
|---|---|
| `roboflow:<workspace>/<project>/<version>` | A Roboflow export. Exporting is free tier. |
| `url:<https://.../archive.zip>` | A hosted zip or tar.gz. |
| `hf:<dataset repo id>` | A Hugging Face dataset repo (needs `HF_TOKEN`). |

Public CC BY 4.0 bullet-hole sets on Roboflow Universe make a decent starting pool. Check the licence on each set and record the exact specs you used, so the run can be repeated. The one-liner below uses placeholder specs; replace them with the sets you choose.

## 6. Docker options and environment

The real values belong in the repo-level `.env` file, which Git ignores. Copy them into the Vast console when you launch. Secrets must never be committed here.

### Run A, YOLO11s baseline

This is the recommended first run. It uses a public pool and skips phase 2 because `FINETUNE_SOURCE` is empty.

```text
-p 8888:8888 -e OPEN_BUTTON_PORT=8888 -e ROBOFLOW_API_KEY=REPLACE_ME -e PRETRAIN_SOURCES=roboflow:your-workspace/your-project/1,roboflow:your-workspace/your-project-2/1 -e FINETUNE_SOURCE= -e HF_TOKEN=REPLACE_ME -e HF_MODEL_REPO=your-user/your-model-repo -e MODEL=yolo11s.pt -e IMGSZ=960 -e EPOCHS=150 -e FINETUNE_EPOCHS=40 -e BATCH=16 -e PATIENCE=40 -e CACHE=ram -e ENABLE_JUPYTER=true -e JUPYTER_TOKEN=REPLACE_ME
```

### Run B, YOLO26n benchmark

Same line as Run A, with two changes:

```text
-e MODEL=yolo26n.pt -e IMGSZ=640
```

Artifacts go into a separate `<stamp>-yolo26n` folder in the Hugging Face model repo, so both runs coexist.

### In-domain fine-tuning run

After real frames have been labelled and published as a Roboflow version, replace the empty setting:

```text
-e FINETUNE_SOURCE=roboflow:<your-ws>/<your-proj>/<ver>
```

Leave `PRETRAIN_SOURCES` alone. Phase 1 pretrains on the pool; phase 2 adapts the model to the actual camera and target.

## 7. Configuration notes

- `PRETRAIN_SOURCES` is a comma-separated list of dataset specs: `roboflow:<ws>/<proj>/<ver>`, `url:<zip/tar.gz>`, or `hf:<dataset repo>`.
- `POOL_DROP_CLASSES` defaults to `contour,target,sticker,shell,casing,hull`. Override it if a new source has a hole class whose name matches one of those fragments.
- An empty `FINETUNE_SOURCE` skips phase 2 and produces the baseline model.
- `IMGSZ` must match the inference resolution. The ONNX export has a static input size.
- `CACHE=ram` is worth using when the dataset fits in memory.
- Only use sources you control or can pin to a fixed version. A dataset in someone else's account can change or disappear without notice.

## 8. After training

1. Inspect `runs/` in JupyterLab.
2. Check `results.png`, `results.csv`, and the confusion matrix.
3. Confirm that `weights/best.pt` and `weights/best.onnx` exist.
4. Confirm the artifacts reached the configured Hugging Face folder.
5. Save the run folder and metrics before destroying the Vast instance.

The upload includes:

```text
best.pt
best.onnx
results.png
results.csv
confusion_matrix.png
confusion_matrix_normalized.png
args.yaml
PR_curve.png
labels.jpg
```

## 9. Datasets to leave out

- Paywalled sets where only preview images are public. You cannot reproduce the run, and the licence may not allow training on them.
- Flatbed scans of printed shooting cards. They do not look like a live camera view of a coloured target through a lens. A paper's scoring method can still be worth reading even when its images are not worth training on.

## 10. GPU offer search

```bash
vastai search offers 'gpu_name in [RTX_3090,RTX_5090] num_gpus=1 verified=true rentable=true direct_port_count>=1' -o dph
```
