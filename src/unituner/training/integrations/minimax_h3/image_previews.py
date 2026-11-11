"""Render native H3-Image flow and restoration previews without moving the training cursor."""

from pathlib import Path

import torch

from unituner.config.training import TrainingConfig
from unituner.data.datasets.restoration import RestorationDataset
from unituner.data.media.image import save_image
from unituner.inference.restoration.vae import encode_restoration_image
from unituner.models.minimax_h3.image import MiniMaxH3ImageTransformer
from unituner.models.minimax_h3.image_cache import read_image_record
from unituner.models.minimax_h3.image_encoders import load_image_encoders
from unituner.models.minimax_h3.image_inference import generate_image
from unituner.training.integrations.minimax_h3.image import MiniMaxH3ImageBinding
from unituner.training.integrations.minimax_h3.restoration import MiniMaxH3ImageRestoration
from unituner.training.previews.context import preview_context
from unituner.training.previews.restoration import RestorationPreviews, _comparison


class MiniMaxH3ImagePreviews:
    """Borrow the active denoiser; own frozen CPU encoders only during rendering."""

    def __init__(self, config: TrainingConfig, dataset: RestorationDataset | None = None):
        if config.preview is None:
            raise ValueError("H3-Image previews require preview settings")
        self.config = config
        self.dataset = dataset
        self.restoration = RestorationPreviews(config, dataset) if dataset is not None else None

    def __call__(
        self,
        update: int,
        model: MiniMaxH3ImageTransformer,
        binding: MiniMaxH3ImageBinding,
        method: MiniMaxH3ImageRestoration | None,
    ) -> None:
        config = self.config
        preview = config.preview
        if preview is None or config.model.vae is None:
            raise ValueError("H3-Image previews require VAE and preview settings")
        device = next(model.parameters()).device
        with preview_context(model, cuda_devices=(device.index or 0,) if device.type == "cuda" else ()):
            if method is None:
                vae, text = load_image_encoders(config.model, device=torch.device("cpu"))
                for index, sample in enumerate(preview.samples):
                    pixels = generate_image(
                        model,
                        vae,
                        text,
                        sample.prompt,
                        width=sample.width if "width" in sample.model_fields_set else None,
                        height=sample.height if "height" in sample.model_fields_set else None,
                        size_policy=sample.h3_size_policy,
                        seed=sample.seed,
                        images=tuple(Path(path) for path in sample.reference_images),
                    )
                    save_image(pixels[0, :, 0].cpu(), self._path(update, index))
            else:
                if self.dataset is None or self.restoration is None:
                    raise ValueError("H3-Image restoration preview needs indexed pairs")
                for index, (_, sample) in enumerate(self.restoration.select(update)):
                    pair = sample.pair
                    low, high = self.dataset.read(
                        pair, sample.resolution, sample.seed, sample.horizontal_flip_probability
                    )
                    with torch.no_grad():
                        low = low.unsqueeze(0).to(device)
                        high = high.unsqueeze(0).to(device)
                        record = read_image_record(pair.cache_directory, pair.caption, device=device)
                        encoded = encode_restoration_image(method.vae, low, posterior="mean")
                        restored = binding.restore(
                            encoded,
                            record.text.unsqueeze(0),
                            record.token_tags.unsqueeze(0),
                            sigma=method.sigma,
                            references=tuple(reference.unsqueeze(0) for reference in record.references),
                        )
                        generated = method.vae.decode(restored)[:, :, 0]
                    save_image(_comparison(low[0], generated[0], high[0]), self._path(update, index))

    def _path(self, update: int, index: int) -> Path:
        return Path(self.config.output.directory) / "previews" / f"step-{update:08d}-{index:02d}.png"
