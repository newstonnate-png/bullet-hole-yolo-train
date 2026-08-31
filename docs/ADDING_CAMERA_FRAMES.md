# Adding real camera frames (the phase-2 fine-tune set)

The first model is usually trained on public photos taken with whatever camera the dataset author had. The biggest gain comes from fine-tuning on frames from the camera you will actually use, pointed at the target you will actually score. That set lives in `datasets/finetune/` and feeds phase 2 of `scripts/train.py`.

## The loop

1. **Capture frames.** Use your camera tool to grab frames of the printed target, with the same crop, rotation and zoom the scoring application will use. Single shots or short bursts both work. Aim for 200 to 400 frames covering:
   - few, medium, and many holes in a frame
   - the lane's real lighting, including glare or shadow when it happens
   - a few holes on or near zone boundaries (useful regression cases for the scorer)
   - a few clean full-target frames with no holes

2. **Upload to Roboflow.** Pick a workspace and project you control. Drag the frames in.

3. **Label.** One class, `bullet-hole`. Box each hole tightly. Centroid precision is what drives correct zone scoring downstream.

4. **Generate a new version.** Minimal preprocessing is fine (auto-orient plus resize-to-fit). Leave augmentation to `train.py`. Note the new version number and keep it with the run.

5. **Retrain.** Relaunch the Vast instance with `-e FINETUNE_SOURCE=roboflow:<workspace>/<project>/<n>` and leave `PRETRAIN_SOURCES` as it is. Phase 1 pretrains on the pool; phase 2 fine-tunes on the new frames. Artifacts land in a fresh `<stamp>-<model>` folder in the HF model repo, so older runs stay put.

6. **Compare.** Check `results.png` and `mAP50` for the fine-tune run, then CPU-test the new `best.onnx` on a held-out real frame.

## After the first model exists

Once you have a trained ONNX, a pre-labelling step can run it over a folder of fresh frames and write YOLO `.txt` pre-labels. Round 2 onwards becomes review-and-correct instead of labelling from scratch.

## Datasets to leave out

- Paywalled sets where only preview images are public. You cannot reproduce the run, and the licence may not allow training on them.
- Flatbed scans of printed shooting cards. They do not look like a live camera view of a coloured target through a lens. A paper's scoring method can still be worth reading even when its images are not worth training on.

## Why Roboflow versioning

`FINETUNE_SOURCE=roboflow:.../<n>` records exactly what a given `best.onnx` was trained on. Exporting a dataset through the API is free. Only hosted inference and Roboflow Train cost money, and this template uses neither.
