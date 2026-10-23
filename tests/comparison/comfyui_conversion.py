"""Audit converted files using an unchanged ComfyUI loader and Kitchen kernels.

Run separately because ComfyUI registers global layout overrides. This does not
install or modify ComfyUI, and reduced matrices do not establish model quality.
"""

import argparse
import importlib.metadata
import json
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import replace
from importlib import import_module as _import_provider
from pathlib import Path
from types import SimpleNamespace as _ProviderNamespace

import torch
from torch.nn import functional as F

from unitunner.artifacts.conversion import convert_weights
from unitunner.artifacts.formats.quantization import FORMATS
from unitunner.artifacts.quantization import read_weight, tensor_json
from unitunner.artifacts.tensors import TensorReader, save_tensors


def main():
    _provider_state0 = _ProviderNamespace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--backend", choices=("auto", "eager"), default="auto")
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfyui_root.resolve()))
    previous = sys.argv
    # Comfy's process-level memory manager need not take ownership of our GPU.
    sys.argv = ["kitchen-conversion-audit", "--cpu"]
    try:
        _provider_state0.ops = _import_provider("comfy.ops")
        import comfy_kitchen as kitchen
        from comfy_kitchen.tensor import QuantizedTensor
    finally:
        sys.argv = previous
    torch.set_num_threads(1)
    torch.manual_seed(817)
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    context = kitchen.use_backend("eager") if args.backend == "eager" else nullcontext()
    budgets = {"nvfp4": 0.25, "convrot_w4a4": 0.3, "convrot_w4a8": 0.25, "asym_w4a8_int8": 0.2, "w6a8_int8": 0.08}
    with context:
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            values = torch.randn(128, 256, dtype=dtype)
            source = args.output_root / f"source-{dtype}.safetensors"
            save_tensors({"weight": values}, source)
            for precision, spec in FORMATS.items():
                if spec.calibrated:
                    continue
                record = {"precision": precision, "source_dtype": str(dtype), "device": args.device}
                try:
                    output = args.output_root / f"{dtype}-{precision}.safetensors"
                    convert_weights(source, output, precision=precision, device=args.device)
                    with TensorReader(output) as reader:
                        tensors = {key: reader.read(key) for key in reader.keys()}
                        canonical = read_weight(
                            "weight", reader.read, tensor_json(tensors["comfy_quant"]), device=torch.device(args.device)
                        )
                    layer = _provider_state0.ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear(
                        256, 128, bias=False, device=args.device
                    )
                    layer.load_state_dict(tensors, strict=True)
                    # Consumer compute dtype is independent of source dtype.
                    canonical = QuantizedTensor(
                        canonical._qdata, canonical._layout_cls, replace(canonical._params, orig_dtype=torch.bfloat16)
                    )
                    with torch.no_grad():
                        actual_weight = layer.weight.dequantize()
                        expected_weight = canonical.dequantize()
                        torch.testing.assert_close(actual_weight, expected_weight, atol=0, rtol=0)
                        inputs = torch.randn(32, 256, device=args.device, dtype=torch.bfloat16)
                        actual = layer(inputs)
                        if layer._full_precision_mm:
                            expected = F.linear(inputs, actual_weight)
                        elif _provider_state0.ops.QUANT_ALGOS[layer.quant_format].get("quantize_input", True):
                            activations = QuantizedTensor.from_float(inputs, layer.layout_type, scale=None)
                            expected = F.linear(activations, layer.weight)
                        else:
                            expected = F.linear(inputs, layer.weight)
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        dense = F.linear(inputs.float(), values.to(args.device).float())
                        relative_l2 = float((actual.float() - dense).norm() / dense.norm())
                        assert torch.isfinite(actual).all()
                        assert relative_l2 < budgets.get(precision, 0.15 if precision == "float8_e5m2" else 0.08), (
                            relative_l2
                        )
                    record.update(
                        status="passed",
                        dequant_max_error=0,
                        consumer_max_error=0,
                        relative_l2_vs_float=relative_l2,
                        full_precision_mm=layer._full_precision_mm,
                    )
                except Exception as error:
                    record.update(status="failed", error=f"{type(error).__name__}: {error}")
                records.append(record)
                print(json.dumps(record), flush=True)
    report = {
        "kitchen_version": importlib.metadata.version("comfy-kitchen"),
        "comfyui_commit": subprocess.check_output(
            ["git", "-C", str(args.comfyui_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "torch": torch.__version__,
        "backend": args.backend,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "scope": "Actual ComfyUI strict Linear loading, dequantization and forward; reduced matrices, not image-quality validation. AWQ/SVDQuant have no current ComfyUI loading branches.",
        "cases": records,
    }
    (args.output_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(record["status"] != "passed" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
