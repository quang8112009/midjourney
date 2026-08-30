import unittest
import warnings
from unittest.mock import patch

from starlette.exceptions import StarletteDeprecationWarning

warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)

from fastapi.testclient import TestClient

from app.api.routes import generation_rate_limiter
from app.core.config import settings
from app.services.model_service import (
    MAX_SEED,
    GenerationCapacityError,
    GenerationError,
    GenerationResult,
    ModelBusyError,
    ModelLoadError,
    ModelUnavailableError,
    OutputSaveError,
)
from main import app


def fake_result(model="stable-diffusion", count=1, seed=42):
    return GenerationResult(
        images=tuple(object() for _ in range(count)),
        model=model,
        seed=seed,
        seeds=tuple(seed + index for index in range(count)),
        elapsed_seconds=0.25,
    )


class GenerateApiTests(unittest.TestCase):
    def setUp(self):
        generation_rate_limiter.reset()
        self.client = TestClient(app)

    @patch(
        "app.api.routes.model_service.save_images",
        return_value=["/tmp/generated.png"],
    )
    @patch(
        "app.api.routes.model_service.generate_image",
        return_value=fake_result(),
    )
    def test_existing_request_uses_typed_reproducible_defaults(self, generate, _save):
        response = self.client.post("/api/v1/generate", json={"prompt": " a red kite "})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["prompt"], "a red kite")
        self.assertEqual(payload["effective_prompt"], "a red kite")
        self.assertFalse(payload["prompt_enhanced"])
        self.assertEqual(payload["model"], "stable-diffusion")
        self.assertEqual(payload["images"], ["/outputs/generated.png"])
        self.assertEqual(payload["parameters"]["seed"], 42)
        self.assertEqual(payload["parameters"]["seeds"], [42])
        self.assertEqual(payload["parameters"]["num_images"], 1)
        self.assertEqual(generate.call_args.kwargs["prompt"], "a red kite")
        self.assertEqual(generate.call_args.kwargs["model"], "stable-diffusion")
        self.assertFalse(generate.call_args.kwargs["wait"])

    @patch(
        "app.api.routes.model_service.save_images",
        return_value=["/tmp/generated.png"],
    )
    @patch(
        "app.api.routes.model_service.generate_image",
        return_value=fake_result(seed=17),
    )
    @patch(
        "app.api.routes.enhance_prompt",
        return_value="a cat, cinematic photography, natural window light, sharp focus",
    )
    def test_prompt_enhancement_is_opt_in_and_uses_request_seed(
        self,
        enhancer,
        generate,
        _save,
    ):
        response = self.client.post(
            "/api/v1/generate",
            json={"prompt": " a cat ", "seed": 17, "enhance_prompt": True},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["prompt"], "a cat")
        self.assertTrue(payload["prompt_enhanced"])
        self.assertEqual(
            payload["effective_prompt"],
            "a cat, cinematic photography, natural window light, sharp focus",
        )
        enhancer.assert_called_once_with("a cat", seed=17)
        self.assertEqual(generate.call_args.kwargs["prompt"], payload["effective_prompt"])
        self.assertEqual(generate.call_args.kwargs["seed"], 17)

    @patch(
        "app.api.routes.model_service.save_images",
        return_value=["/tmp/generated.png"],
    )
    @patch(
        "app.api.routes.model_service.generate_image",
        return_value=fake_result(seed=73),
    )
    @patch("app.api.routes.enhance_prompt", return_value="a cat, enhanced")
    @patch("app.api.routes.secrets.randbits", return_value=73)
    def test_unseeded_enhancement_shares_resolved_seed_with_inference(
        self,
        random_seed,
        enhancer,
        generate,
        _save,
    ):
        response = self.client.post(
            "/api/v1/generate",
            json={"prompt": "a cat", "enhance_prompt": True},
        )

        self.assertEqual(response.status_code, 200)
        random_seed.assert_called_once_with(63)
        enhancer.assert_called_once_with("a cat", seed=73)
        self.assertEqual(generate.call_args.kwargs["seed"], 73)
        self.assertEqual(response.json()["parameters"]["seed"], 73)

    @patch(
        "app.api.routes.model_service.save_images",
        return_value=["/tmp/a.png", "/tmp/b.png"],
    )
    @patch("app.api.routes.model_service.generate_image")
    def test_pixart_defaults_and_multi_image_seeds(self, generate, _save):
        generate.return_value = fake_result("pixart-alpha", count=2, seed=9)
        response = self.client.post(
            "/api/v1/generate",
            json={"model": "pixart-alpha", "prompt": "castle", "num_images": 2},
        )
        self.assertEqual(response.status_code, 200)
        parameters = response.json()["parameters"]
        self.assertEqual((parameters["width"], parameters["height"]), (512, 512))
        self.assertEqual(parameters["guidance_scale"], 4.5)
        self.assertEqual(parameters["seeds"], [9, 10])

    def test_validation_boundaries_and_unknown_fields(self):
        invalid_requests = [
            ({"prompt": "   "}, "prompt"),
            ({"prompt": "x" * (settings.MAX_PROMPT_LENGTH + 1)}, "string_too_long"),
            (
                {
                    "prompt": "x",
                    "negative_prompt": "x" * (settings.MAX_NEGATIVE_PROMPT_LENGTH + 1),
                },
                "string_too_long",
            ),
            ({"prompt": "x", "seed": MAX_SEED + 1}, "seed"),
            ({"prompt": "x", "width": 2048, "height": 2048}, "pixel limit"),
            ({"prompt": "x", "width": 1024, "height": 256}, "aspect"),
            (
                {
                    "prompt": "x",
                    "width": 1024,
                    "height": 1024,
                    "num_inference_steps": 100,
                },
                "too expensive",
            ),
            (
                {"model": "pixart-alpha", "prompt": "x", "width": 1024, "height": 1024},
                "512x512",
            ),
            ({"prompt": "x", "num_inference_step": 20}, "extra_forbidden"),
        ]
        for payload, expected in invalid_requests:
            with self.subTest(payload=payload):
                response = self.client.post("/api/v1/generate", json=payload)
                self.assertEqual(response.status_code, 422)
                self.assertIn(expected, response.text)

    def test_error_statuses_are_stable_and_sanitized(self):
        cases = [
            (ModelBusyError("internal busy"), 429, "worker is busy"),
            (ModelUnavailableError("CUDA unavailable"), 503, "CUDA unavailable"),
            (ModelLoadError("Unable to load checkpoint"), 503, "Unable to load"),
            (GenerationCapacityError("GPU memory exceeded"), 503, "GPU memory"),
            (OutputSaveError("could not be saved"), 507, "could not be saved"),
            (GenerationError("Image inference failed."), 500, "Image inference failed"),
        ]
        for error, status, detail in cases:
            generation_rate_limiter.reset()
            with (
                self.subTest(error=type(error).__name__),
                patch("app.api.routes.model_service.generate_image", side_effect=error),
            ):
                response = self.client.post("/api/v1/generate", json={"prompt": "test"})
                self.assertEqual(response.status_code, status)
                self.assertIn(detail, response.json()["detail"])

    @patch(
        "app.api.routes.model_service.generate_image",
        side_effect=RuntimeError("secret filesystem path"),
    )
    def test_unexpected_error_does_not_leak_details(self, _generate):
        with self.assertLogs("app.api.routes", level="ERROR"):
            response = self.client.post("/api/v1/generate", json={"prompt": "test"})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("secret filesystem path", response.text)
        self.assertIn("Request ID", response.text)

    @patch(
        "app.api.routes.model_service.save_images",
        return_value=["/tmp/generated.png"],
    )
    @patch(
        "app.api.routes.model_service.generate_image",
        return_value=fake_result(),
    )
    def test_output_path_is_not_derived_from_host_header(self, _generate, _save):
        response = self.client.post(
            "/api/v1/generate",
            json={"prompt": "test"},
            headers={"Host": "attacker.example"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["images"], ["/outputs/generated.png"])
        self.assertNotIn("attacker.example", response.text)

    @patch(
        "app.api.routes.model_service.save_images",
        return_value=["/tmp/generated.png"],
    )
    @patch(
        "app.api.routes.model_service.generate_image",
        return_value=fake_result(),
    )
    def test_rate_limit_returns_retry_after(self, _generate, _save):
        original_limit = generation_rate_limiter.limit
        generation_rate_limiter.limit = 2
        try:
            self.assertEqual(
                self.client.post("/api/v1/generate", json={"prompt": "one"}).status_code,
                200,
            )
            self.assertEqual(
                self.client.post("/api/v1/generate", json={"prompt": "two"}).status_code,
                200,
            )
            response = self.client.post("/api/v1/generate", json={"prompt": "three"})
        finally:
            generation_rate_limiter.limit = original_limit
            generation_rate_limiter.reset()
        self.assertEqual(response.status_code, 429)
        self.assertIn("Retry-After", response.headers)

    def test_sample_is_post_only(self):
        self.assertEqual(self.client.get("/api/v1/generate/sample").status_code, 405)

    @patch(
        "app.api.routes.model_service.save_images",
        return_value=["/tmp/sample.png"],
    )
    @patch(
        "app.api.routes.model_service.generate_image",
        return_value=fake_result(),
    )
    @patch("app.api.routes.FileResponse")
    def test_sample_post_uses_unique_bounded_generation(self, file_response, generate, save):
        file_response.return_value = {"ok": True}
        response = self.client.post("/api/v1/generate/sample")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(generate.call_args.kwargs["width"], 512)
        self.assertEqual(generate.call_args.kwargs["num_inference_steps"], 20)
        generation_id = save.call_args.args[1]
        self.assertTrue(generation_id.startswith("sample_"))

    def test_api_v1_health_endpoint(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "healthy")
        self.assertIn("busy", payload)
        self.assertIn("cuda_available", payload)
        self.assertIn("device", payload)
        self.assertIn("model_loaded", payload)

    @patch(
        "app.api.routes.model_service.generate_image",
        side_effect=ModelUnavailableError(
            "PixArt-Alpha requires a CUDA GPU. Use Stable Diffusion on CPU"
        ),
    )
    def test_pixart_on_cpu_raises_503(self, _generate):
        response = self.client.post(
            "/api/v1/generate",
            json={"prompt": "a portrait", "model": "pixart-alpha"},
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("PixArt-Alpha requires a CUDA GPU", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
