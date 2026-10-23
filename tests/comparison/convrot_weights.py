"""Compare published INT8 ConvRot weight encoding with unchanged Musubi quantization."""

import argparse
import json
import subprocess
import sys
from importlib import import_module as _import_provider
from pathlib import Path
from types import SimpleNamespace as _ProviderNamespace

import torch

from unitunner.artifacts.quantization import quantize_weight


def main():
    _provider_api1 = _ProviderNamespace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musubi-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.musubi_root / "src"))
    import comfy_kitchen

    _provider_api1.quantize_int8_convrot_weight = _import_provider(
        "musubi_tuner.modules.convrot_int8_kernels"
    ).quantize_int8_convrot_weight

    torch.set_num_threads(1)
    records = []
    for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            torch.manual_seed(61)
            values = torch.randn(64, 256, device=device, dtype=dtype)
            with comfy_kitchen.use_backend("eager"):
                packed = quantize_weight(values, "int8_convrot")
            codes, scales = _provider_api1.quantize_int8_convrot_weight(values, 256)
            torch.testing.assert_close(packed._qdata, codes, atol=0, rtol=0)
            torch.testing.assert_close(packed._params.scale, scales, atol=0, rtol=0)
            records.append({"device": device, "dtype": str(dtype), "codes_exact": True, "scales_exact": True})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "musubi_commit": subprocess.check_output(
                    ["git", "-C", str(args.musubi_root), "rev-parse", "HEAD"], text=True
                ).strip(),
                "quantization_backend": "eager",
                "scope": "Weight encoding only. UniTunner uses floating compute and does not implement Musubi dynamic activation quantization.",
                "cases": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"passed": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
