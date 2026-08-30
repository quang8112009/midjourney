import io
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app.api.routes import generation_rate_limiter
from app.core.config import settings
from app.services.model_service import GenerationResult, ModelBusyError
from main import app


def encoded_image(
    *,
    image_format="PNG",
    size=(256, 256),
    mode="RGB",
    color="white",
    exif=None,
):
    output = io.BytesIO()
    image = Image.new(mode, size, color)
    kwargs = {"format": image_format}
    if exif is not None:
        kwargs["exif"] = exif
    image.save(output, **kwargs)
    return output.getvalue()


def masked_png(*, size=(256, 256), alpha=False):
    if alpha:
        image = Image.new("RGBA", size, (255, 255, 255, 0))
        for y in range(64, 192):
            for x in range(64, 192):
                image.putpixel((x, y), (255, 255, 255, 255))
    else:
        image = Image.new("L", size, 0)
        for y in range(64, 192):
            for x in range(64, 192):
                image.putpixel((x, y), 255)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def fake_edit_result(seed=42):
    return GenerationResult(
        images=(object(),),
        model="stable-diffusion",
        seed=seed,
        seeds=(seed,),
        elapsed_seconds=0.4,
    )


class EditApiTests(unittest.TestCase):
    def setUp(self):
        generation_rate_limiter.reset()
        self.client = TestClient(app)
        self.source = encoded_image(image_format="WEBP", color="navy")
        self.mask = masked_png(alpha=True)

    def _post(self, *, prompt="recolor the shirt to red", source=None, mask=None, data=None):
        form = {"prompt": prompt, **(data or {})}
        source_payload = self.source if source is None else source
        mask_payload = self.mask if mask is None else mask
        files = {
            "source_image": ("source.webp", source_payload, "image/webp"),
            "mask": ("mask.png", mask_payload, "image/png"),
        }
        return self.client.post("/api/v1/edit", data=form, files=files)

    @patch("app.api.routes.model_service.save_images", return_value=["/tmp/edited.png"])
    @patch("app.api.routes.model_service.edit_image", return_value=fake_edit_result(seed=17))
    def test_success_decodes_files_and_returns_intent_and_parameters(self, edit, _save):
        response = self._post(
            prompt="  recolor the shirt to red  ",
            data={
                "negative_prompt": "  blurry  ",
                "num_inference_steps": "8",
                "guidance_scale": "6.25",
                "strength": "0.75",
                "seed": "17",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["images"], ["/outputs/edited.png"])
        self.assertEqual(body["prompt"], "recolor the shirt to red")
        self.assertEqual(body["model"], "stable-diffusion")
        self.assertEqual(
            body["intent"],
            {
                "status": "ok",
                "action": "recolor",
                "target": "shirt",
                "attribute": "red",
                "scope": "local",
                "assumption": None,
            },
        )
        self.assertEqual(body["parameters"]["width"], 256)
        self.assertEqual(body["parameters"]["height"], 256)
        self.assertEqual(body["parameters"]["num_inference_steps"], 8)
        self.assertEqual(body["parameters"]["guidance_scale"], 6.25)
        self.assertEqual(body["parameters"]["strength"], 0.75)
        self.assertEqual(body["parameters"]["seed"], 17)

        kwargs = edit.call_args.kwargs
        self.assertEqual(kwargs["negative_prompt"], "blurry")
        self.assertEqual(kwargs["model"], "stable-diffusion")
        self.assertFalse(kwargs["wait"])
        self.assertEqual(kwargs["source_image"].mode, "RGB")
        self.assertEqual(kwargs["mask_image"].mode, "L")
        self.assertEqual(kwargs["mask_image"].getpixel((0, 0)), 0)
        self.assertEqual(kwargs["mask_image"].getpixel((128, 128)), 255)

    @patch("app.api.routes.model_service.save_images", return_value=["/tmp/edited.png"])
    @patch("app.api.routes.model_service.edit_image", return_value=fake_edit_result())
    def test_defaults_use_source_dimensions_and_stable_diffusion_policy(self, edit, _save):
        source = encoded_image(size=(512, 256), image_format="JPEG", color="green")
        response = self._post(source=source, mask=masked_png(size=(512, 256)))

        self.assertEqual(response.status_code, 200, response.text)
        parameters = response.json()["parameters"]
        self.assertEqual((parameters["width"], parameters["height"]), (512, 256))
        self.assertEqual(parameters["num_inference_steps"], settings.DEFAULT_STEPS)
        self.assertEqual(parameters["guidance_scale"], settings.DEFAULT_GUIDANCE_SCALE)
        self.assertEqual(edit.call_args.kwargs["strength"], 1.0)

    @patch("app.api.routes.model_service.edit_image")
    def test_clarifying_and_compound_prompts_are_structured_422s(self, edit):
        cases = [
            ("change the shirt", "edit_clarification_required", None),
            (
                "recolor the shirt to red and blur the background",
                "compound_edit_not_supported",
                2,
            ),
        ]
        for prompt, code, count in cases:
            generation_rate_limiter.reset()
            with self.subTest(prompt=prompt):
                response = self._post(prompt=prompt)
                self.assertEqual(response.status_code, 422)
                detail = response.json()["detail"]
                self.assertEqual(detail["code"], code)
                if count is not None:
                    self.assertEqual(detail["instruction_count"], count)
        edit.assert_not_called()

    @patch("app.api.routes.model_service.edit_image")
    def test_rejects_unsupported_or_invalid_encoded_images(self, edit):
        gif_source = encoded_image(image_format="GIF")
        jpeg_mask = encoded_image(image_format="JPEG", mode="L")
        cases = [
            ({"source": gif_source}, "unsupported_edit_image_format"),
            ({"mask": jpeg_mask}, "unsupported_edit_image_format"),
            ({"source": b"not an image"}, "invalid_edit_image"),
        ]
        for inputs, code in cases:
            generation_rate_limiter.reset()
            with self.subTest(code=code):
                response = self._post(**inputs)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()["detail"]["code"], code)
        edit.assert_not_called()

    @patch("app.api.routes.model_service.edit_image")
    def test_rejects_mismatched_empty_and_invalid_geometry(self, edit):
        cases = [
            (
                {"mask": masked_png(size=(256, 264))},
                {},
                "edit_mask_size_mismatch",
            ),
            (
                {"mask": encoded_image(mode="L", color=0)},
                {},
                "empty_edit_mask",
            ),
            (
                {
                    "source": encoded_image(size=(250, 256)),
                    "mask": masked_png(size=(250, 256)),
                },
                {},
                "invalid_edit_dimensions",
            ),
            (
                {
                    "source": encoded_image(size=(1024, 1024)),
                    "mask": masked_png(size=(1024, 1024)),
                },
                {"num_inference_steps": "100"},
                "edit_request_too_expensive",
            ),
        ]
        for inputs, data, code in cases:
            generation_rate_limiter.reset()
            with self.subTest(code=code):
                response = self._post(data=data, **inputs)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertEqual(response.json()["detail"]["code"], code)
        edit.assert_not_called()

    @patch("app.api.routes.model_service.edit_image")
    def test_uploads_are_bounded_per_file(self, edit):
        with patch.object(settings, "MAX_EDIT_UPLOAD_BYTES", 32):
            response = self._post()
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"]["code"], "edit_upload_too_large")
        edit.assert_not_called()

    @patch("app.api.routes.model_service.save_images", return_value=["/tmp/edited.png"])
    @patch("app.api.routes.model_service.edit_image", return_value=fake_edit_result())
    def test_edit_uses_the_shared_generation_rate_limit(self, _edit, _save):
        original_limit = generation_rate_limiter.limit
        generation_rate_limiter.limit = 1
        try:
            self.assertEqual(self._post().status_code, 200)
            limited = self._post()
        finally:
            generation_rate_limiter.limit = original_limit
            generation_rate_limiter.reset()
        self.assertEqual(limited.status_code, 429)
        self.assertIn("Retry-After", limited.headers)

    @patch(
        "app.api.routes.model_service.edit_image",
        side_effect=ModelBusyError("internal worker state"),
    )
    def test_edit_reuses_public_model_error_mapping(self, _edit):
        response = self._post()
        self.assertEqual(response.status_code, 429)
        self.assertIn("worker is busy", response.json()["detail"])
        self.assertEqual(response.headers["Retry-After"], "2")

    def test_missing_required_multipart_fields_are_rejected(self):
        response = self.client.post("/api/v1/edit", data={"prompt": "recolor the shirt red"})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
