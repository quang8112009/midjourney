# Diffusion Transformer research and datasets

## Recommended architecture path

The original [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) replaces the latent-diffusion U-Net with a transformer over latent patches. Its published checkpoint is class-conditioned on ImageNet labels, so it is valuable as the architectural foundation but is not a direct fit for this project's natural-language prompt API.

[PixArt-Alpha](https://arxiv.org/abs/2310.00426) adds text cross-attention and a staged training strategy for text-to-image synthesis. The selected [`PixArt-alpha/PixArt-XL-2-512x512`](https://huggingface.co/PixArt-alpha/PixArt-XL-2-512x512) checkpoint has an official Diffusers pipeline and is the smallest practical integration target for this service.

Useful follow-up designs are:

- [PixArt-Sigma](https://arxiv.org/abs/2403.04692), which extends weak-to-strong training to higher resolutions.
- [Stable Diffusion 3](https://arxiv.org/abs/2403.03206), which combines rectified flow with a multimodal diffusion transformer that uses separate image/text weights and bidirectional information flow.
- The [official Diffusers PixArt pipeline](https://huggingface.co/docs/diffusers/api/pipelines/pixart) for the supported inference interface.

## Dataset decision table

| Dataset | Best use here | Important restriction |
|---|---|---|
| [Public Domain 12M](https://huggingface.co/datasets/Spawning/PD12M) | Preferred starting point for future image-caption fine-tuning | Images are identified as public-domain/CC0 and synthetically captioned, but provenance should still be audited before production use. |
| [ImageNet-1K](https://www.image-net.org/download.php) | Reproducing or benchmarking original class-conditioned DiT | Access terms limit use to non-commercial research and education; labels are classes rather than prompts. |
| [JourneyDB](https://journeydb.github.io/) | Research evaluation of generated-image prompt/style understanding | Customized terms prohibit commercial use and competitive research against Midjourney or Discord. Do not train the product on it. |
| [SAM-LLaVA Captions](https://huggingface.co/datasets/PixArt-alpha/SAM-LLaVA-Captions10M) | Studying PixArt's dense-caption strategy | The published artifact is primarily captions/URLs; evaluate the underlying SA-1B image license separately. |
| [Re-LAION-5B](https://laion.ai/blog/relaion-5b/) | Large-scale research experiments | It is URL/metadata based, can contain unsafe material, and the linked images retain their own copyrights; LAION advises against unreviewed industrial use. |
| [DataComp](https://www.datacomp.ai/dcclip/getting_started.html) | Research on web-scale filtering and data quality | Metadata is CC-BY-4.0, while individual linked images retain separate copyrights. |

For a product-oriented fine-tuning phase, start with a small, reviewed PD12M subset plus properly licensed first-party images. Store source URL, creator/license, license evidence, caption provenance, content-safety result, perceptual hash, and opt-out status for every item. Do not treat a metadata license as a license to the linked image.

## Deferred fine-tuning path

Training from scratch is not appropriate for this repository: PixArt-Alpha reports hundreds of A100 GPU-days even with its efficiency improvements. A later phase should use transformer LoRA against the pretrained checkpoint, an `image`/`text` dataset schema, fixed validation prompts, checkpointed GPU training, and CLIP/aesthetic metrics supplemented by human review. The official [PixArt training repository](https://github.com/PixArt-alpha/PixArt-alpha) includes LoRA and custom dataset examples.

No dataset is downloaded and no training code is included in the current inference release.
