"""Reproducible visual audit: PYTHONPATH=src python tests/data/preview_degradation.py --help."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from unitunner.data.degradation.config import DegradationConfig
from unitunner.data.degradation.pipeline import degrade_image


def baseline(config: DegradationConfig) -> DegradationConfig:
    """Official fixed-pixel distributions using this CPU codec implementation."""
    data = config.model_dump()
    data["resolution_exponent"] = 0.0
    data["high_scale_protection"] = False
    for name in ("first", "second"):
        data[name]["noise"]["grain_probability"] = 0.0
        data[name]["compression"]["probabilities"] = (1.0, 0.0, 0.0)
    return DegradationConfig.model_validate(data)


def tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255).permute(2, 0, 1)


def picture(value: torch.Tensor) -> Image.Image:
    return Image.fromarray((value.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    rows, records = [], []
    for index, path in enumerate(args.sources):
        source = ImageOps.fit(Image.open(path).convert("RGB"), (1024, 1024), method=Image.Resampling.LANCZOS)
        source.save(args.output / f"source_{index}.png")
        for scale in (4, 6, 8):
            seed = 13 + index
            cfg = DegradationConfig(scales=(float(scale),))
            old_trace, trace, before_trace = [], [], []
            high = tensor(source)
            reference = degrade_image(high, baseline(cfg), seed, trace=old_trace)
            before = degrade_image(
                high, cfg.model_copy(update={"high_scale_protection": False}), seed, trace=before_trace
            )
            result = degrade_image(high, cfg, seed, trace=trace)
            assert result.shape == high.shape and torch.isfinite(result).all()
            native = degrade_image(high, cfg.model_copy(update={"output_mode": "native"}), seed)
            images = [source, picture(reference), picture(result), picture(native)]
            stem = f"source_{index}_x{scale}_seed{seed}"
            for suffix, im in zip(("hq", "baseline", "adapted", "native"), images, strict=True):
                im.save(args.output / f"{stem}_{suffix}.png")
            picture(before).save(args.output / f"{stem}_before.png")
            pair = Image.new("RGB", (3072, 1060), "#dddddd")
            pair_draw = ImageDraw.Draw(pair)
            for column, (title, im) in enumerate(
                zip(
                    ("HQ", "Fixed official parameters (CPU codecs)", "After: resolution + scale protection"),
                    (source, picture(reference), picture(result)),
                    strict=True,
                )
            ):
                pair_draw.text((column * 1024 + 8, 8), f"{title} / x{scale} / seed {seed}", fill="black")
                pair.paste(im, (column * 1024, 36))
            pair.save(args.output / f"{stem}_comparison.png")
            rows.append((f"image {index + 1} / x{scale} / seed {seed}", images))
            records.append(
                {"sample": stem, "baseline_trace": old_trace, "before_trace": before_trace, "adapted_trace": trace}
            )
    tile, label = 280, 35
    canvas = Image.new("RGB", (4 * tile, len(rows) * (tile + label)), "#dddddd")
    draw = ImageDraw.Draw(canvas)
    for row, (name, images) in enumerate(rows):
        y = row * (tile + label)
        for column, (title, im) in enumerate(
            zip(
                ("HQ", "Fixed official parameters*", "Resolution + codec/grain", "Native LQ (nearest zoom)"),
                images,
                strict=True,
            )
        ):
            x = column * tile
            draw.text((x + 4, y + 2), title, fill="black")
            draw.text((x + 4, y + 17), name, fill="black")
            canvas.paste(
                im.resize((tile, tile), Image.Resampling.NEAREST if column == 3 else Image.Resampling.LANCZOS),
                (x, y + label),
            )
    canvas.save(args.output / "comparison.png")
    # Cross-resolution smoke checks, including fractional and extreme scales.
    source = Image.open(args.sources[0]).convert("RGB")
    for size in (256, 512, 1024, 1280, 2048):
        high = tensor(ImageOps.fit(source, (size, size)))
        for scale in (1.0, 2.5, 4.0, 8.0, 16.0):
            trace = []
            result = degrade_image(high, DegradationConfig(scales=(scale,)), 19, trace=trace)
            assert result.shape == high.shape and torch.isfinite(result).all()
            assert next(e for e in trace if e["op"] == "native_lq")["shape"] == [int(size / scale)] * 2
            records.append({"resolution": size, "scale": scale, "trace": trace})
    (args.output / "traces.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} visual samples and 25 resolution/scale checks to {args.output}")
    print("*Fixed official parameter distribution via CPU OpenCV codecs, not pixel-exact GPU DiffJPEG.")


if __name__ == "__main__":
    main()
