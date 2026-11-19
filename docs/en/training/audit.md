# Native training audit

Enable `audit = true` under `[output]`, or pass `--output.audit true`. The default is false for compatibility. Native observation points record parameter installation, noise grids and publisher keys without monkey-patching the sampler, optimizer or CLI. Help and argument failures retain their native behavior.

The run's `training-audit/` directory contains resolved configuration, runtime arguments and resume/output paths, a versioned manifest, actual targets with algorithm/rank/alpha and trainable parameter/category/refiner counts, native flow sigma grids, exact publisher keys, verified exports and a separate resume-file inventory. Restoration marks flow controls inapplicable and records midtimestep, explicit sigma, calibration and posterior settings.

Exports use native `weights/` paths and latest aliases. Stage weights cannot substitute for final weights. Dual-expert runs require both final experts. Adapter keys come from the native LoRA/LoHa/LoKr and ComfyUI/Diffusers publishers, including model-owned fused layouts. Full-model and encoder files outside that publisher are explicitly marked `header_recorded` rather than claiming adapter-key verification. Tensor payloads are not loaded for export auditing.

Resume comparisons reuse the numerical config identity from `build_resume_identity`, excluding output/log paths and publication schedules while retaining layer selection, rank, effective sampling, optimizer and dataset configuration. A new output directory can locate the prior audit through checkpoint parents. Legacy checkpoints and notebook-wrapper manifests defer to native restore's actual dataset/model validation instead of comparing incompatible fingerprints.

Only model references are recorded: no new model-file inventory, existence checks or hash scans are added. Existing checkpoint validation, model preparation, downloads, default layer selection, expert routing, noise distributions and RNG consumption are preserved.
