"""Declare shared H3 generation and preview inputs without model or media imports."""

from pydantic import BaseModel, ConfigDict, Field

from unituner.models.minimax_h3.config.media import MiniMaxH3ControlInput, MiniMaxH3ReferenceInput
from unituner.models.minimax_h3.config.sampling import MiniMaxH3SamplingConfig


class MiniMaxH3InputsConfig(BaseModel):
    """Choose a caption and explicit native anchors, indexed controls or ordered references.

    Video FL2VA uses first/last anchors. Single-frame FL2VA uses controls with
    explicit native24Hz pixel-frame coordinates; order and duplicate coordinates
    are retained. Relative paths resolve from the caller's working directory.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    prompt: str = Field(min_length=1)
    first_frame: str | None = Field(default=None, min_length=1)
    last_frame: str | None = Field(default=None, min_length=1)
    controls: tuple[MiniMaxH3ControlInput, ...] = Field(default=(), max_length=1000)
    target_index: int | None = Field(default=None, ge=0)
    references: tuple[MiniMaxH3ReferenceInput, ...] = Field(default=(), max_length=12)


def validate_minimax_h3_inputs(inputs: MiniMaxH3InputsConfig, sampling: MiniMaxH3SamplingConfig, *, task: str) -> None:
    """Reject unsupported task/condition combinations without probing files or loading models."""
    if not inputs.prompt.strip():
        raise ValueError("MiniMaxH3 generation prompt must contain non-whitespace text")
    single = sampling.frames == 1
    if not single and (inputs.target_index is not None or inputs.controls):
        raise ValueError("MiniMaxH3 target_index/controls require single-frame generation")
    if task == "t2va":
        if inputs.first_frame is not None or inputs.last_frame is not None or inputs.controls or inputs.references:
            raise ValueError("MiniMaxH3 T2VA does not accept conditions or references")
    elif task == "fl2va":
        if inputs.references:
            raise ValueError("MiniMaxH3 FL2VA does not accept references")
        if single:
            if not inputs.controls or inputs.first_frame is not None or inputs.last_frame is not None:
                raise ValueError("MiniMaxH3 single-frame FL2VA requires indexed controls without first/last aliases")
        elif inputs.first_frame is None and inputs.last_frame is None:
            raise ValueError("MiniMaxH3 video FL2VA requires first_frame and/or last_frame")
    elif task == "ref2va":
        if inputs.first_frame is not None or inputs.last_frame is not None or inputs.controls:
            raise ValueError("MiniMaxH3 Ref2VA does not accept FL anchors/controls")
        images = sum(reference.kind == "image" for reference in inputs.references)
        videos = sum(reference.kind == "video" for reference in inputs.references)
        if images > 9 or videos > 3 or not images + videos:
            raise ValueError("MiniMaxH3 Ref2VA requires visual references, at most9images/3videos")
        if single and any(reference.kind == "audio" for reference in inputs.references):
            raise ValueError("MiniMaxH3 single-frame generation rejects standalone audio references")
    else:
        raise ValueError("MiniMaxH3 generation task must be t2va/fl2va/ref2va")


class MiniMaxH3SampleConfig(BaseModel):
    """Pair native sampling choices with the conditions of one generated AV sample."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    inputs: MiniMaxH3InputsConfig
    sampling: MiniMaxH3SamplingConfig = Field(default_factory=MiniMaxH3SamplingConfig)
