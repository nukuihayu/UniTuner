"""Map explicit Diffusers convolutional VAE keys to native FLUX artifact paths."""

import re


def native_autoencoder_key(key: str, stages: int) -> str:
    """Map Diffusers VAE paths to native keys, reversing decoder stage indices."""
    if key.startswith("quant_conv."):
        return "encoder." + key
    if key.startswith("post_quant_conv."):
        return "decoder." + key
    key = key.replace(".conv_norm_out.", ".norm_out.")
    key = key.replace(".mid_block.resnets.0.", ".mid.block_1.").replace(".mid_block.resnets.1.", ".mid.block_2.")
    key = key.replace(".mid_block.attentions.0.", ".mid.attn_1.")
    for source, target in (
        ("group_norm", "norm"),
        ("to_q", "q"),
        ("to_k", "k"),
        ("to_v", "v"),
        ("to_out.0", "proj_out"),
    ):
        key = key.replace(f".attn_1.{source}.", f".attn_1.{target}.")
    key = re.sub(r"encoder\.down_blocks\.(\d+)\.resnets\.", r"encoder.down.\1.block.", key)
    key = re.sub(r"encoder\.down_blocks\.(\d+)\.downsamplers\.0\.", r"encoder.down.\1.downsample.", key)
    key = re.sub(r"decoder\.up_blocks\.(\d+)\.", lambda match: f"decoder.up.{stages - 1 - int(match[1])}.", key)
    key = key.replace(".resnets.", ".block.").replace(".upsamplers.0.", ".upsample.")
    return key.replace(".conv_shortcut.", ".nin_shortcut.")
