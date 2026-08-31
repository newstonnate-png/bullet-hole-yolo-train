# Model choice and resolution notes

## Which YOLO

| Model | COCO mAP50-95 | CPU-ONNX @640 | Params | Notes |
|---|---|---|---|---|
| yolo11n | 39.5 | ~56 ms | 2.6 M | fastest; use if CPU latency is tight |
| **yolo11s** | **47.0** | ~90 ms | 9.4 M | **default** — best accuracy/effort for small holes on a CPU-only scoring PC |
| yolo11m | 51.5 | ~183 ms | 20 M | only if the scoring PC has real headroom |
| yolo26n | 40.9 | ~39 ms | 2.4 M | newest line; NMS-free; ProgLoss + STAL specifically target small objects; ~31–43 % faster CPU |
| yolo26s | 48.6 | ~87 ms | 9.5 M | same accuracy as 11s, faster; validate before trusting (new) |

Those CPU figures are Ultralytics' reference numbers. The machine doing the scoring will be slower. With frame skipping in the consumer and a static target, 2–4 FPS is enough.

Start with `yolo11s`: it exports everywhere and has been around long enough to trust. Then run a second job with `MODEL=yolo26n.pt` and compare `mAP50` and real-frame CPU latency. Keep the winner for your scoring app.

## Two phases

`scripts/train.py` trains on `datasets/pool/` first (pooled public data, a general idea of what a hole looks like). If `data/finetune.yaml` exists, phase 2 continues from the phase-1 weights on `datasets/finetune/` (real zoom-lens frames of the printed target). Phase 2 uses fewer epochs (`FINETUNE_EPOCHS`, default 40), `lr0=0.002`, and `freeze=10`, so the backbone holds still and only the head adapts to this camera.

The first run has no fine-tune set, so phase 2 is skipped. That baseline is good enough to wire up your scoring app while you collect real frames. See `ADDING_CAMERA_FRAMES.md`.

## Input resolution

Bullet holes are small, so resolution matters more than model size. Default is `IMGSZ=960`. If the holes are tiny in the cropped zoom-lens frame and the GPU has VRAM, try `IMGSZ=1280`. Keep training and inference `imgsz` equal: the ONNX export bakes the size in with `dynamic=False`.

## Export

`train.py` exports `best.onnx` with `opset=12`, `simplify=True`, `dynamic=False`. That combination has the widest support in `onnxruntime` on the scoring PC. NCNN is an option later if the range moves to an ARM board.
