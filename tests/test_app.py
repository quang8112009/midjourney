import unittest
import uuid
import warnings
from pathlib import Path
from unittest.mock import patch

from starlette.exceptions import StarletteDeprecationWarning

warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)

from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from main import app


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_health_readiness_and_openapi(self):
        root = self.client.get("/")
        health = self.client.get("/health")
        ready = self.client.get("/ready")
        schema = self.client.get("/openapi.json")

        self.assertEqual(root.status_code, 200)
        self.assertEqual(root.json()["version"], "1.3.0")
        self.assertEqual(root.json()["edit"], "/api/v1/edit")
        self.assertIn("pixart-alpha", root.json()["models"])
        self.assertEqual(health.status_code, 200)
        self.assertIn("busy", health.json())
        self.assertEqual(ready.status_code, 200)
        self.assertIn("/api/v1/generate", schema.json()["paths"])
        self.assertIn("/api/v1/edit", schema.json()["paths"])
        self.assertIn("/api/v1/chat", schema.json()["paths"])
        self.assertIn("/ready", schema.json()["paths"])
        self.assertNotIn("ReasoningDecision", schema.json()["components"]["schemas"])

    def test_readiness_reports_configured_cuda_failure(self):
        fake_status = {
            "model_loaded": False,
            "active_model": None,
            "loading_model": None,
            "configured_device": "cuda",
            "device": "cpu",
            "cuda_available": False,
            "busy": False,
            "last_load_error": None,
            "generation_count": 0,
            "last_generation_seconds": None,
        }
        with (
            patch.object(settings, "DEVICE", "cuda"),
            patch("main.model_service.status", return_value=fake_status),
        ):
            response = self.client.get("/ready")
        self.assertEqual(response.status_code, 503)
        self.assertIn("CUDA", response.text)

    def test_static_output_is_retrievable(self):
        filename = f"test_{uuid.uuid4().hex}.png"
        path = Path(settings.OUTPUT_DIR) / filename
        Image.new("RGB", (4, 4), "purple").save(path, format="PNG")
        try:
            response = self.client.get(f"/outputs/{filename}")
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")

    def test_frontend_is_served(self):
        response = self.client.get("/ui/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("pixart-alpha", response.text)
        self.assertIn('id="enhancePrompt"', response.text)
        self.assertIn("enhance_prompt: enhancePromptToggle.checked", response.text)
        self.assertIn('id="promptTemplate"', response.text)
        for category in ("portrait", "landscape", "product-shot", "anime", "concept-art"):
            self.assertIn(f'<option value="{category}">', response.text)
        self.assertIn("const PROMPT_TEMPLATES = Object.freeze", response.text)
        self.assertIn("window.confirm", response.text)
        self.assertIn("lastAppliedTemplate", response.text)
        self.assertIn('id="chatForm"', response.text)
        self.assertIn("sessionStorage", response.text)
        self.assertIn("`${API_BASE}/chat`", response.text)

    def test_cors_preflight(self):
        origin = (
            "http://example.test"
            if settings.ALLOWED_ORIGINS == ["*"]
            else settings.ALLOWED_ORIGINS[0]
        )
        response = self.client.options(
            "/api/v1/generate",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("access-control-allow-origin", response.headers)


if __name__ == "__main__":
    unittest.main()
