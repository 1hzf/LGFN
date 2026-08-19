#!/usr/bin/env python3
"""Inference for the RGB-only LGFN-B2 deployment checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train_rgb import (
    IMAGE_SUFFIXES,
    LGFNB2RGBDeploy,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class InferenceDataset(Dataset):
    def __init__(self, input_dir, image_size):
        self.paths = sorted(
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise RuntimeError(f"No RGB images found: {input_dir}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            tensor = self.transform(image)
        return tensor, path.stem, height, width


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = LGFNB2RGBDeploy(backbone_weights=None)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = InferenceDataset(args.input_dir, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    completed = 0
    skipped = 0
    with torch.inference_mode():
        for batch_index, (rgb, names, heights, widths) in enumerate(loader, start=1):
            rgb = rgb.to(device, non_blocking=True)
            mask_logits, _ = model(rgb)
            for index, name in enumerate(names):
                output_path = args.output_dir / f"{name}.png"
                if output_path.exists() and not args.overwrite:
                    skipped += 1
                    continue
                resized = F.interpolate(
                    mask_logits[index : index + 1],
                    size=(int(heights[index]), int(widths[index])),
                    mode="bilinear",
                    align_corners=False,
                )
                prediction = torch.sigmoid(resized)[0, 0].cpu().numpy()
                prediction = np.clip(
                    np.rint(prediction * 255.0), 0, 255
                ).astype(np.uint8)
                Image.fromarray(prediction, mode="L").save(output_path)
                completed += 1
            if batch_index % 10 == 0 or batch_index == len(loader):
                print(
                    f"[lgfn-infer] {batch_index}/{len(loader)} "
                    f"written={completed} skipped={skipped}",
                    flush=True,
                )
    print(
        f"[lgfn-infer] complete samples={len(dataset)} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
