#!/usr/bin/env python
"""
WMAR Watermark Detection CLI.

This script is called via subprocess from the main waterwipe environment
to detect watermarks in images using the WMAR encoder.

Usage:
    cd vendors/wmar
    uv run python wmar_detect.py \
        --image path/to/image.png \
        --model taming \
        --checkpoints-dir checkpoints \
        --sync-path checkpoints/syncmodel.jit.pt \
        --original-codes path/to/codes.npy \
        --wm-delta 2.0 \
        --wm-gamma 0.5
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SYNCSEAL_URL = "https://dl.fbaipublicfiles.com/wmar/syncseal/paper/syncmodel.jit.pt"


def ensure_syncseal_checkpoint(sync_path: Path) -> None:
    """Download syncseal checkpoint if it doesn't exist."""
    if sync_path.exists():
        return

    print(f"SyncSeal checkpoint not found at {sync_path}", file=sys.stderr)
    print(f"Downloading from {SYNCSEAL_URL}...", file=sys.stderr)

    sync_path.parent.mkdir(parents=True, exist_ok=True)

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rDownloading: {percent}%", end="", file=sys.stderr, flush=True)

    urllib.request.urlretrieve(SYNCSEAL_URL, sync_path, reporthook=progress_hook)
    print(file=sys.stderr)  # newline after progress
    print(f"Downloaded SyncSeal checkpoint to {sync_path}", file=sys.stderr)


from wmar.models.taming_wrapper import TamingARMMWrapper
from wmar.models.chameleon_wrapper import ChameleonARMMWrapper
from wmar.models.rar_wrapper import RarARMMWrapper
from wmar.watermarking.gentime_watermark import (
    GentimeWatermark,
    SeedStrategy,
    SplitStrategy,
)
from wmar.watermarking.synchronization import SyncManager


MODEL_PATHS = {
    "taming": "2021-04-03T19-39-50_cin_transformer",
    "chameleon7b": "Anole-7b-v0.1",
    "rar": "rar",
}


def load_model(model_name: str, checkpoints_dir: Path, seed: int = 42):
    """Load the specified model."""
    model_subdir = MODEL_PATHS.get(model_name, model_name)
    modelpath = str(checkpoints_dir / model_subdir)

    if model_name == "taming":
        return TamingARMMWrapper(modelpath)
    elif model_name == "chameleon7b":
        return ChameleonARMMWrapper(modelpath, seed)
    elif model_name == "rar":
        return RarARMMWrapper(modelpath)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to tensor in [-1, 1] range."""
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # HWC -> CHW
    img_tensor = img_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
    return img_tensor


def detect_watermark(
    image_path: Path,
    model_name: str,
    checkpoints_dir: Path,
    sync_path: Path,
    original_codes_path: Path | None,
    wm_delta: float,
    wm_gamma: float,
) -> dict:
    """
    Detect watermark in an image.

    Returns:
        dict with keys:
            - p_value: statistical significance of watermark detection
            - detected: boolean, True if p_value < 0.01
            - extracted_codes: list of extracted token codes
            - l0_distance: fraction of codes that differ from original (if provided)
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_tensor = pil_to_tensor(img).unsqueeze(0).cuda()  # [1, 3, H, W]

    # Load model
    model = load_model(model_name, checkpoints_dir)
    vocab_size = model.get_total_vocab_size()

    # Setup watermarker for detection
    watermarker = GentimeWatermark(
        model.get_vq(),
        vocab_size,
        SeedStrategy.FIXED,
        SplitStrategy.RANDOM_STRATIFIED,
        context_size=0,
        delta=wm_delta,
        gamma=wm_gamma,
        device=model.device,
    )
    model.set_watermarker(watermarker)

    # Setup sync manager to remove synchronization
    sync_manager = SyncManager(str(sync_path), device="cuda")

    # Remove sync to align the image
    with torch.no_grad():
        img_aligned = sync_manager.remove_sync(img_tensor)

        # Extract codes from aligned image
        extracted_codes = model.images_to_codes(img_aligned)

        # Detect watermark
        p_value = watermarker.detect(extracted_codes)

    extracted_codes_np = extracted_codes.cpu().numpy().flatten()

    result = {
        "p_value": float(p_value),
        "detected": p_value < 0.01,
        "extracted_codes": extracted_codes_np.tolist(),
    }

    # Compute L0 distance if original codes provided
    if original_codes_path is not None and original_codes_path.exists():
        original_codes = np.load(original_codes_path).flatten()
        if len(original_codes) == len(extracted_codes_np):
            l0_distance = float(np.mean(original_codes != extracted_codes_np))
            result["l0_distance"] = l0_distance
            result["codes_match_count"] = int(np.sum(original_codes == extracted_codes_np))
            result["codes_total"] = len(original_codes)

    return result


def main():
    parser = argparse.ArgumentParser(description="Detect WMAR watermark in an image")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the image to detect",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["taming", "chameleon7b", "rar"],
        help="Model to use for detection",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--sync-path",
        type=Path,
        default=Path("checkpoints/syncmodel.jit.pt"),
        help="Path to syncseal checkpoint",
    )
    parser.add_argument(
        "--original-codes",
        type=Path,
        default=None,
        help="Path to original codes .npy file (optional, for L0 computation)",
    )
    parser.add_argument(
        "--wm-delta",
        type=float,
        default=2.0,
        help="Watermark delta (must match generation)",
    )
    parser.add_argument(
        "--wm-gamma",
        type=float,
        default=0.5,
        help="Watermark gamma (must match generation)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: stdout)",
    )

    args = parser.parse_args()

    # Ensure syncseal checkpoint exists (download if needed)
    ensure_syncseal_checkpoint(args.sync_path)

    if not args.image.exists():
        print(json.dumps({"error": f"Image not found: {args.image}"}))
        sys.exit(1)

    try:
        result = detect_watermark(
            image_path=args.image,
            model_name=args.model,
            checkpoints_dir=args.checkpoints_dir,
            sync_path=args.sync_path,
            original_codes_path=args.original_codes,
            wm_delta=args.wm_delta,
            wm_gamma=args.wm_gamma,
        )

        output_json = json.dumps(result)

        if args.output:
            with open(args.output, "w") as f:
                f.write(output_json)
        else:
            print(output_json)

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
