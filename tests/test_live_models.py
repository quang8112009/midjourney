"""Optional checkpoint-backed smoke tests; disabled in the normal CI lane."""

import os
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from app.core.config import settings
from app.services.model_service import PIXART_ALPHA, ModelService


class LiveModelSmokeTests(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("RUN_LIVE_SD_TESTS") == "1",
        "set RUN_LIVE_SD_TESTS=1 to download/run Stable Diffusion",
    )
    def test_stable_diffusion_fixed_seed_smoke(self):
        service = ModelService()
        try:
            with patch.object(settings, "DEVICE", "auto"):
                result = service.generate_image(
                    prompt="a red circle on a white background",
                    width=256,
                    height=256,
                    num_inference_steps=1,
                    guidance_scale=1.0,
                    seed=7,
                )
            self.assertEqual(len(result.images), 1)
            self.assertEqual(result.images[0].size, (256, 256))
            self.assertEqual(result.seed, 7)
        finally:
            service.unload_model()

    @unittest.skipUnless(
        os.getenv("RUN_LIVE_SD_EDIT_TESTS") == "1",
        "set RUN_LIVE_SD_EDIT_TESTS=1 to download/run Stable Diffusion inpainting",
    )
    def test_stable_diffusion_inpainting_fixed_seed_smoke(self):
        service = ModelService()
        source = Image.new("RGB", (256, 256), "white")
        mask = Image.new("L", source.size, 0)
        mask.paste(255, (64, 64, 192, 192))
        try:
            with patch.object(settings, "DEVICE", "auto"):
                result = service.edit_image(
                    prompt="a red circle",
                    source_image=source,
                    mask_image=mask,
                    num_inference_steps=1,
                    guidance_scale=1.0,
                    strength=1.0,
                    seed=7,
                )
            self.assertEqual(len(result.images), 1)
            self.assertEqual(result.images[0].size, source.size)
            self.assertEqual(result.seed, 7)
        finally:
            service.unload_model()

    @unittest.skipUnless(
        os.getenv("RUN_LIVE_PIXART_TESTS") == "1" and torch.cuda.is_available(),
        "set RUN_LIVE_PIXART_TESTS=1 on CUDA to download/run PixArt",
    )
    def test_pixart_fixed_seed_smoke(self):
        service = ModelService()
        try:
            with (
                patch.object(settings, "DEVICE", "cuda"),
                patch.object(settings, "DTYPE", "auto"),
            ):
                result = service.generate_image(
                    prompt="a red circle on a white background",
                    model=PIXART_ALPHA,
                    num_inference_steps=1,
                    guidance_scale=1.0,
                    seed=7,
                )
            self.assertEqual(len(result.images), 1)
            self.assertEqual(result.images[0].size, (512, 512))
            self.assertEqual(result.seed, 7)
        finally:
            service.unload_model()


if __name__ == "__main__":
    unittest.main()
