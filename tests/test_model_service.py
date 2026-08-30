import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from app.core.config import settings
from app.services.model_service import (
    INPAINTING_TASK,
    MAX_SEED,
    PIXART_ALPHA,
    STABLE_DIFFUSION,
    TEXT_TO_IMAGE_TASK,
    GenerationCapacityError,
    GenerationError,
    ModelBusyError,
    ModelLoadError,
    ModelService,
    ModelUnavailableError,
    OutputSaveError,
    ResolvedGenerationDefaults,
    resolve_generation_defaults,
)


class FakePipeline:
    def __init__(self, call_handler=None):
        self.scheduler = SimpleNamespace(config={"name": "original"})
        self.to_device = None
        self.offloaded = False
        self.vae_slicing = False
        self.last_call = None
        self.call_handler = call_handler

    def to(self, device):
        self.to_device = device
        return self

    def enable_model_cpu_offload(self):
        self.offloaded = True

    def enable_vae_slicing(self):
        self.vae_slicing = True

    def __call__(self, **kwargs):
        self.last_call = kwargs
        if self.call_handler:
            return self.call_handler(**kwargs)
        count = kwargs["num_images_per_prompt"]
        return SimpleNamespace(images=[f"image-{index}" for index in range(count)])


class BrokenImage:
    def save(self, *_args, **_kwargs):
        raise OSError("disk write failed")


class ModelServiceTests(unittest.TestCase):
    def test_generation_defaults_are_frozen_shared_and_preserve_overrides(self):
        with (
            patch.object(settings, "DEFAULT_WIDTH", 640),
            patch.object(settings, "DEFAULT_HEIGHT", 384),
            patch.object(settings, "DEFAULT_STEPS", 17),
            patch.object(settings, "DEFAULT_GUIDANCE_SCALE", 6.25),
            patch.object(settings, "PIXART_DEFAULT_STEPS", 23),
            patch.object(settings, "PIXART_DEFAULT_GUIDANCE_SCALE", 4.0),
        ):
            stable = resolve_generation_defaults(STABLE_DIFFUSION)
            pixart = resolve_generation_defaults(
                PIXART_ALPHA,
                num_inference_steps=9,
                guidance_scale=5.5,
            )

        self.assertIsInstance(stable, ResolvedGenerationDefaults)
        self.assertEqual((stable.width, stable.height), (640, 384))
        self.assertEqual(stable.num_inference_steps, 17)
        self.assertEqual(stable.guidance_scale, 6.25)
        self.assertEqual((pixart.width, pixart.height), (512, 512))
        self.assertEqual(pixart.num_inference_steps, 9)
        self.assertEqual(pixart.guidance_scale, 5.5)
        with self.assertRaises(FrozenInstanceError):
            stable.width = 512

    def test_cpu_generation_is_reproducible_and_loads_once(self):
        pipeline = FakePipeline()
        pipeline_loader = MagicMock(return_value=pipeline)
        scheduler = object()
        service = ModelService(
            pipeline_loader=pipeline_loader,
            scheduler_loader=MagicMock(return_value=scheduler),
        )

        with (
            patch.object(settings, "DEVICE", "cpu"),
            patch.object(settings, "DTYPE", "float16"),
            patch("app.services.model_service.torch.cuda.is_available", return_value=False),
        ):
            result = service.generate_image(prompt="test", seed=123, num_images=2)
            service.load_model(STABLE_DIFFUSION)

        self.assertEqual(result.images, ("image-0", "image-1"))
        self.assertEqual(result.seed, 123)
        self.assertEqual(result.seeds, (123, 124))
        self.assertEqual(service.active_model, STABLE_DIFFUSION)
        self.assertEqual(pipeline.to_device, "cpu")
        self.assertIs(pipeline.scheduler, scheduler)
        self.assertEqual(
            [generator.initial_seed() for generator in pipeline.last_call["generator"]],
            [123, 124],
        )
        self.assertEqual(pipeline.last_call["width"], settings.DEFAULT_WIDTH)
        self.assertEqual(pipeline.last_call["height"], settings.DEFAULT_HEIGHT)
        self.assertEqual(pipeline.last_call["num_inference_steps"], settings.DEFAULT_STEPS)
        self.assertEqual(pipeline.last_call["guidance_scale"], settings.DEFAULT_GUIDANCE_SCALE)
        self.assertIs(pipeline_loader.call_args.kwargs["torch_dtype"], torch.float32)
        pipeline_loader.assert_called_once()

    def test_edit_uses_inpainting_loader_and_explicit_source_dimensions(self):
        inpainting_pipeline = FakePipeline()
        inpainting_loader = MagicMock(return_value=inpainting_pipeline)
        text_loader = MagicMock()
        service = ModelService(
            pipeline_loader=text_loader,
            inpainting_pipeline_loader=inpainting_loader,
        )
        source = Image.new("RGBA", (320, 256), (30, 60, 90, 255))
        mask = Image.new("RGB", source.size, "black")
        mask.paste("white", (80, 64, 240, 192))

        with (
            patch.object(settings, "DEVICE", "cpu"),
            patch("app.services.model_service.torch.cuda.is_available", return_value=False),
        ):
            result = service.edit_image(
                prompt="  replace the sign  ",
                negative_prompt="  blurry  ",
                source_image=source,
                mask_image=mask,
                num_inference_steps=7,
                guidance_scale=5.0,
                strength=0.65,
                seed=37,
            )

        text_loader.assert_not_called()
        inpainting_loader.assert_called_once_with(
            settings.MODEL_ID,
            torch_dtype=torch.float32,
            use_safetensors=True,
            cache_dir=settings.MODEL_CACHE_DIR,
        )
        call = inpainting_pipeline.last_call
        self.assertEqual(call["prompt"], "replace the sign")
        self.assertEqual(call["negative_prompt"], "blurry")
        self.assertEqual(call["image"].mode, "RGB")
        self.assertEqual(call["mask_image"].mode, "L")
        self.assertEqual((call["width"], call["height"]), source.size)
        self.assertEqual(call["num_inference_steps"], 7)
        self.assertEqual(call["guidance_scale"], 5.0)
        self.assertEqual(call["strength"], 0.65)
        self.assertEqual(call["num_images_per_prompt"], 1)
        self.assertEqual(call["generator"].initial_seed(), 37)
        self.assertEqual(result.seed, 37)
        self.assertEqual(result.seeds, (37,))
        self.assertEqual(len(result.images), 1)
        self.assertEqual(service.active_task, INPAINTING_TASK)
        self.assertEqual(service.status()["generation_count"], 1)

    def test_model_task_identity_reloads_only_when_identity_changes(self):
        first_text_pipeline = FakePipeline()
        second_text_pipeline = FakePipeline()
        inpainting_pipeline = FakePipeline()
        text_loader = MagicMock(side_effect=[first_text_pipeline, second_text_pipeline])
        inpainting_loader = MagicMock(return_value=inpainting_pipeline)
        scheduler_loader = MagicMock(return_value=object())
        service = ModelService(
            pipeline_loader=text_loader,
            scheduler_loader=scheduler_loader,
            inpainting_pipeline_loader=inpainting_loader,
        )
        source = Image.new("RGB", (256, 256), "blue")
        mask = Image.new("L", source.size, 255)

        with (
            patch.object(settings, "DEVICE", "cpu"),
            patch("app.services.model_service.torch.cuda.is_available", return_value=False),
        ):
            service.generate_image(prompt="first", seed=1)
            service.edit_image(
                prompt="edit one",
                source_image=source,
                mask_image=mask,
                seed=2,
            )
            service.edit_image(
                prompt="edit two",
                source_image=source,
                mask_image=mask,
                seed=3,
            )
            service.generate_image(prompt="second", seed=4)

        self.assertEqual(text_loader.call_count, 2)
        self.assertEqual(inpainting_loader.call_count, 1)
        self.assertEqual(scheduler_loader.call_count, 2)
        self.assertIs(service.pipe, second_text_pipeline)
        self.assertEqual(service.active_model, STABLE_DIFFUSION)
        self.assertEqual(service.active_task, TEXT_TO_IMAGE_TASK)
        self.assertEqual(service.status()["generation_count"], 4)

    def test_edit_direct_call_validation_rejects_invalid_inputs_before_loading(self):
        loader = MagicMock()
        service = ModelService(inpainting_pipeline_loader=loader)
        source = Image.new("RGB", (256, 256), "blue")
        mask = Image.new("L", source.size, 255)
        blank_mask = Image.new("L", source.size, 0)
        mismatched_mask = Image.new("L", (256, 264), 255)
        invalid_size_source = Image.new("RGB", (257, 256), "blue")
        invalid_size_mask = Image.new("L", invalid_size_source.size, 255)
        base = {
            "prompt": "replace it",
            "source_image": source,
            "mask_image": mask,
        }
        invalid_calls = [
            {**base, "prompt": "   "},
            {**base, "model": PIXART_ALPHA},
            {**base, "source_image": object()},
            {**base, "mask_image": object()},
            {**base, "mask_image": blank_mask},
            {**base, "mask_image": mismatched_mask},
            {
                **base,
                "source_image": invalid_size_source,
                "mask_image": invalid_size_mask,
            },
            {**base, "num_inference_steps": 0},
            {**base, "guidance_scale": 0.5},
            {**base, "strength": 0.0},
            {**base, "seed": MAX_SEED + 1},
        ]

        for kwargs in invalid_calls:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                service.edit_image(**kwargs)
        loader.assert_not_called()

    def test_random_seed_is_resolved_and_returned(self):
        service = ModelService(
            pipeline_loader=MagicMock(return_value=FakePipeline()),
            scheduler_loader=MagicMock(return_value=object()),
        )
        with (
            patch.object(settings, "DEVICE", "cpu"),
            patch("app.services.model_service.secrets.randbits", return_value=456),
            patch("app.services.model_service.torch.cuda.is_available", return_value=False),
        ):
            result = service.generate_image(prompt="test")
        self.assertEqual(result.seed, 456)
        self.assertEqual(result.seeds, (456,))

    def test_seed_wraps_safely_for_multi_image_batch(self):
        service = ModelService(
            pipeline_loader=MagicMock(return_value=FakePipeline()),
            scheduler_loader=MagicMock(return_value=object()),
        )
        with (
            patch.object(settings, "DEVICE", "cpu"),
            patch("app.services.model_service.torch.cuda.is_available", return_value=False),
        ):
            result = service.generate_image(prompt="test", seed=MAX_SEED, num_images=2)
        self.assertEqual(result.seeds, (MAX_SEED, 0))

    def test_pixart_rejection_preserves_loaded_pipeline(self):
        service = ModelService()
        existing_pipeline = FakePipeline()
        service.pipe = existing_pipeline
        service.active_model = STABLE_DIFFUSION
        with (
            patch.object(settings, "DEVICE", "cpu"),
            patch("app.services.model_service.torch.cuda.is_available", return_value=False),
        ):
            with self.assertRaises(ModelUnavailableError):
                service.load_model(PIXART_ALPHA)
        self.assertIs(service.pipe, existing_pipeline)
        self.assertEqual(service.active_model, STABLE_DIFFUSION)

    def test_switch_to_pixart_uses_cuda_offload(self):
        stable_pipeline = FakePipeline()
        pixart_pipeline = FakePipeline()
        loader = MagicMock(side_effect=[stable_pipeline, pixart_pipeline])
        service = ModelService(
            pipeline_loader=loader,
            scheduler_loader=MagicMock(return_value=object()),
        )
        with (
            patch.object(settings, "DEVICE", "cuda"),
            patch.object(settings, "DTYPE", "auto"),
            patch.object(settings, "MODEL_CPU_OFFLOAD", True),
            patch("app.services.model_service.torch.cuda.is_available", return_value=True),
            patch(
                "app.services.model_service.torch.cuda.is_bf16_supported",
                return_value=True,
            ),
            patch("app.services.model_service.torch.cuda.empty_cache"),
        ):
            service.load_model(STABLE_DIFFUSION)
            service.load_model(PIXART_ALPHA)

        self.assertEqual(service.active_model, PIXART_ALPHA)
        self.assertIs(service.pipe, pixart_pipeline)
        self.assertTrue(pixart_pipeline.offloaded)
        self.assertIs(loader.call_args.kwargs["torch_dtype"], torch.bfloat16)

    def test_pixart_backend_resolves_its_own_defaults(self):
        pipeline = FakePipeline()
        service = ModelService(pipeline_loader=MagicMock(return_value=pipeline))
        with (
            patch.object(settings, "DEVICE", "cuda"),
            patch.object(settings, "DTYPE", "auto"),
            patch.object(settings, "MODEL_CPU_OFFLOAD", True),
            patch("app.services.model_service.torch.cuda.is_available", return_value=True),
            patch("app.services.model_service.torch.cuda.is_bf16_supported", return_value=True),
            patch("app.services.model_service.torch.cuda.empty_cache"),
        ):
            service.generate_image(prompt="  castle  ", model=PIXART_ALPHA, seed=5)

        self.assertEqual(pipeline.last_call["prompt"], "castle")
        self.assertEqual(pipeline.last_call["width"], 512)
        self.assertEqual(pipeline.last_call["height"], 512)
        self.assertEqual(pipeline.last_call["num_inference_steps"], settings.PIXART_DEFAULT_STEPS)
        self.assertEqual(
            pipeline.last_call["guidance_scale"], settings.PIXART_DEFAULT_GUIDANCE_SCALE
        )

    def test_unsupported_explicit_bfloat16_is_rejected_before_loading(self):
        loader = MagicMock()
        service = ModelService(pipeline_loader=loader)
        with (
            patch.object(settings, "DEVICE", "cuda"),
            patch.object(settings, "DTYPE", "bfloat16"),
            patch("app.services.model_service.torch.cuda.is_available", return_value=True),
            patch("app.services.model_service.torch.cuda.is_bf16_supported", return_value=False),
        ):
            with self.assertRaisesRegex(ModelUnavailableError, "bfloat16"):
                service.load_model()
        loader.assert_not_called()

    def test_load_failure_is_sanitized_and_recorded(self):
        service = ModelService(pipeline_loader=MagicMock(side_effect=OSError("secret")))
        with (
            patch.object(settings, "DEVICE", "cpu"),
            patch("app.services.model_service.torch.cuda.is_available", return_value=False),
        ):
            with self.assertLogs("app.services.model_service", level="ERROR"):
                with self.assertRaisesRegex(ModelLoadError, "Unable to load") as raised:
                    service.load_model()
        self.assertNotIn("secret", str(raised.exception))
        self.assertFalse(service.is_loaded)
        self.assertIsNone(service.loading_model)
        self.assertEqual(service.status()["last_load_error"], "Unable to load stable-diffusion.")

    def test_generation_and_editing_share_one_non_queueing_slot(self):
        entered = threading.Event()
        release = threading.Event()

        def blocking_call(**kwargs):
            entered.set()
            release.wait(timeout=5)
            return SimpleNamespace(images=["image"])

        service = ModelService(
            pipeline_loader=MagicMock(return_value=FakePipeline(blocking_call)),
            scheduler_loader=MagicMock(return_value=object()),
        )
        source = Image.new("RGB", (256, 256), "blue")
        mask = Image.new("L", source.size, 255)
        with (
            patch.object(settings, "DEVICE", "cpu"),
            patch("app.services.model_service.torch.cuda.is_available", return_value=False),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            future = executor.submit(service.generate_image, prompt="first")
            self.assertTrue(entered.wait(timeout=2))
            self.assertTrue(service.status()["busy"])
            with self.assertRaises(ModelBusyError):
                service.edit_image(
                    prompt="second",
                    source_image=source,
                    mask_image=mask,
                    wait=False,
                )
            release.set()
            future.result(timeout=2)
        self.assertFalse(service.status()["busy"])

    def test_cuda_oom_has_stable_capacity_error(self):
        def out_of_memory(**_kwargs):
            raise torch.cuda.OutOfMemoryError("secret allocator state")

        service = ModelService(
            pipeline_loader=MagicMock(return_value=FakePipeline(out_of_memory)),
            scheduler_loader=MagicMock(return_value=object()),
        )
        with (
            patch.object(settings, "DEVICE", "cpu"),
            patch("app.services.model_service.torch.cuda.is_available", return_value=False),
        ):
            with self.assertRaisesRegex(GenerationCapacityError, "GPU memory"):
                service.generate_image(prompt="test")
        self.assertFalse(service.is_loaded)

    def test_wrong_image_count_is_rejected(self):
        pipeline = FakePipeline(lambda **_kwargs: SimpleNamespace(images=[]))
        service = ModelService(
            pipeline_loader=MagicMock(return_value=pipeline),
            scheduler_loader=MagicMock(return_value=object()),
        )
        with (
            patch.object(settings, "DEVICE", "cpu"),
            patch("app.services.model_service.torch.cuda.is_available", return_value=False),
        ):
            with self.assertRaisesRegex(GenerationError, "unexpected number"):
                service.generate_image(prompt="test")

    def test_internal_argument_validation(self):
        service = ModelService()
        invalid_calls = [
            {"prompt": "   "},
            {"prompt": "x", "seed": MAX_SEED + 1},
            {"prompt": "x", "width": 2048, "height": 2048},
            {"prompt": "x", "width": 1024, "height": 256},
            {"prompt": "x", "model": PIXART_ALPHA, "width": 512, "height": 256},
        ]
        for kwargs in invalid_calls:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                service.generate_image(**kwargs)

    def test_atomic_batch_save_and_count_retention(self):
        service = ModelService()
        image = Image.new("RGB", (8, 8), "red")
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(settings, "OUTPUT_DIR", directory),
                patch.object(settings, "MAX_OUTPUT_FILES", 2),
                patch.object(settings, "MAX_OUTPUT_BYTES", 10_000_000),
                patch.object(settings, "MAX_OUTPUT_AGE_SECONDS", 0),
            ):
                service.save_images([image], "batch_a")
                service.save_images([image], "batch_b")
                paths = service.save_images([image], "batch_c")
                files = sorted(Path(directory).glob("*.png"))

        self.assertEqual(len(files), 2)
        self.assertFalse(any(path.name.startswith("batch_a") for path in files))
        self.assertTrue(paths[0].endswith("batch_c_0.png"))

    def test_partial_batch_failure_rolls_back(self):
        service = ModelService()
        good = Image.new("RGB", (8, 8), "blue")
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(settings, "OUTPUT_DIR", directory),
                patch.object(settings, "MAX_OUTPUT_AGE_SECONDS", 0),
            ):
                with self.assertLogs("app.services.model_service", level="ERROR"):
                    with self.assertRaises(OutputSaveError):
                        service.save_images([good, BrokenImage()], "rollback")
                self.assertEqual(list(Path(directory).glob("*")), [])

    def test_age_and_byte_retention_remove_old_outputs(self):
        service = ModelService()
        image = Image.new("RGB", (8, 8), "orange")
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(settings, "OUTPUT_DIR", directory),
                patch.object(settings, "MAX_OUTPUT_FILES", 100),
                patch.object(settings, "MAX_OUTPUT_BYTES", 10_000_000),
                patch.object(settings, "MAX_OUTPUT_AGE_SECONDS", 10),
            ):
                old_path = Path(service.save_images([image], "old")[0])
                old_time = time.time() - 60
                os.utime(old_path, (old_time, old_time))
                service.save_images([image], "fresh")
                self.assertFalse(old_path.exists())

            with (
                patch.object(settings, "OUTPUT_DIR", directory),
                patch.object(settings, "MAX_OUTPUT_FILES", 100),
                patch.object(settings, "MAX_OUTPUT_BYTES", 100),
                patch.object(settings, "MAX_OUTPUT_AGE_SECONDS", 0),
            ):
                first = Path(service.save_images([image], "bytes_a")[0])
                second = Path(service.save_images([image], "bytes_b")[0])
                self.assertFalse(first.exists())
                self.assertTrue(second.exists())

    def test_output_name_cannot_escape_directory(self):
        service = ModelService()
        image = Image.new("RGB", (8, 8), "green")
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(settings, "OUTPUT_DIR", directory),
        ):
            with self.assertRaises(ValueError):
                service.save_images([image], "../escape")
            with self.assertRaises(ValueError):
                service.save_image(image, "/tmp/escape.png")


if __name__ == "__main__":
    unittest.main()
