"""Exercise selected objectives through public training, previews, export and exact resume."""

import pytest
import torch

from tests.application.test_qwen_image_21_flow import test_public_training_three_updates_and_exact_resume as public_case
from tests.application.test_qwen_image_21_restoration import (
    test_qwen21_restoration_shared_runner_resume_preview_and_export as restoration_case,
)
from unituner.execution.allocator_bootstrap import vbar_initialized


def test_public_qwen_wavelet_full_updates_and_resume(tmp_path):
    public_case(tmp_path, "full", 0, wavelet=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_public_qwen_lora_wavelet_updates_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("UNITUNER_TEST_DEVICE", "cuda")
    public_case(tmp_path, "lora", 2, precision="bf16", wavelet=True)


@pytest.mark.skipif(not vbar_initialized(), reason="Run the VBAR driver before importing Torch")
def test_public_qwen_vbar_wavelet_updates_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("UNITUNER_TEST_DEVICE", "cuda")
    public_case(tmp_path, "lora", 2, "vbar", precision="bf16", wavelet=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_public_qwen_full_wavelet_updates_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("UNITUNER_TEST_DEVICE", "cuda")
    public_case(tmp_path, "full", 0, precision="bf16", wavelet=True)


def test_public_qwen_restoration_wavelet_updates_and_resume(tmp_path):
    restoration_case(tmp_path, "full", wavelet=True)
