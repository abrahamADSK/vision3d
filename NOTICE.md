# NOTICE

Vision3D — FastAPI server for 3D model generation
Copyright (c) 2026 Abraham (abrahamADSK)

This project itself is licensed under the MIT License — see the `LICENSE`
file in the root of this repository for the full text.

Vision3D integrates and depends on a number of third-party components, each
with its own license and terms. The MIT license of Vision3D applies only to
the code written in this repository. Third-party components are governed by
their respective upstream licenses, which you must review and comply with
when using, modifying, or redistributing Vision3D — especially in commercial
contexts.

This file is informational. It summarizes the non-trivial licensing
obligations that downstream users of Vision3D should be aware of. It is not
legal advice and does not override the upstream licenses.

---

## Third-party components with restrictive terms

These components have licenses that impose meaningful obligations beyond a
simple permissive license. Review the upstream license before commercial
use or redistribution.

### Hunyuan3D-2 (Tencent)

- **Upstream**: https://github.com/Tencent/Hunyuan3D-2
- **License**: Tencent Hunyuan Community License (see upstream repository)
- **Usage in Vision3D**: shape generation (`Hunyuan3DDiTFlowMatchingPipeline`),
  texture painting (`Hunyuan3DPaintPipeline`), delighting
  (`hunyuan3d-delight-v2-0`), and the `hy3dgen` Python package. Model weights
  downloaded by `install.sh` into `hf_models/hunyuan3d-*` are subject to the
  same license.
- **Key points**: The Tencent Hunyuan Community License is not a standard
  OSI license. It places restrictions on commercial use, redistribution,
  and derivative works. Review the upstream license text before deploying
  Vision3D for any purpose beyond personal research.

### SDXL Turbo (Stability AI)

- **Upstream**: https://huggingface.co/stabilityai/sdxl-turbo
- **License**: Stability AI Non-Commercial Research Community License
- **Usage in Vision3D**: text-to-image step of the text-to-3D pipeline.
  Weights are auto-downloaded by `diffusers` on first use.
- **Key points**: Non-commercial use only per the upstream license. If you
  need a commercially-usable text-to-image model, swap SDXL Turbo for a
  different one in `_load_t2i_pipeline()`.

### PyMeshLab

- **Upstream**: https://github.com/cnr-isti-vclab/PyMeshLab
- **License**: GPL-3.0
- **Usage in Vision3D**: optional mesh processing helpers (installed by
  `install.sh` as part of the "mesh extras" step).
- **Key points**: PyMeshLab is GPL-3.0. Linking against or redistributing a
  closed-source binary that bundles PyMeshLab can trigger GPL obligations
  on the combined work. For users running Vision3D locally or as a network
  service this is generally not an issue (network use is not
  redistribution), but it matters if you ship Vision3D as an embedded
  component of a closed-source product. If you need to avoid GPL exposure
  entirely, remove `pymeshlab` from the `MESH_EXTRAS` array in `install.sh`
  and accept the reduced mesh-processing capability.

---

## Third-party components with permissive licenses

The following components use standard permissive licenses (MIT, BSD,
Apache 2.0). Attribution is appreciated; there are no meaningful
restrictions on commercial use or redistribution.

| Component | License | Upstream |
|-----------|---------|----------|
| FastAPI | MIT | https://github.com/tiangolo/fastapi |
| Uvicorn | BSD 3-Clause | https://github.com/encode/uvicorn |
| sse-starlette | BSD 3-Clause | https://github.com/sysid/sse-starlette |
| python-multipart | Apache 2.0 | https://github.com/andrew-d/python-multipart |
| Pillow | MIT-CMU (HPND) | https://github.com/python-pillow/Pillow |
| NumPy | BSD 3-Clause | https://github.com/numpy/numpy |
| Trimesh | MIT | https://github.com/mikedh/trimesh |
| pyfqmr | MIT | https://github.com/Kramer84/pyfqmr-Fast-Quadric-Mesh-Reduction |
| rembg | MIT | https://github.com/danielgatis/rembg |
| onnxruntime | MIT | https://github.com/microsoft/onnxruntime |
| diffusers | Apache 2.0 | https://github.com/huggingface/diffusers |
| transformers | Apache 2.0 | https://github.com/huggingface/transformers |
| accelerate | Apache 2.0 | https://github.com/huggingface/accelerate |
| PyTorch | BSD-style | https://github.com/pytorch/pytorch |
| xatlas | MIT | https://github.com/jpcy/xatlas |
| einops | MIT | https://github.com/arogozhnikov/einops |
| omegaconf | BSD 3-Clause | https://github.com/omry/omegaconf |

---

## Trademarks

Trademarks mentioned in this repository are the property of their
respective owners. Their use here is nominative (identifying the products
and projects involved) and does not imply endorsement.

- "Hunyuan" is a trademark of Tencent.
- "Stability AI" and "SDXL" are trademarks of Stability AI Ltd.
- "Apple", "Apple Silicon", "macOS", and "MPS" (Metal Performance Shaders)
  are trademarks of Apple Inc.
- "NVIDIA", "CUDA", and "GeForce" are trademarks of NVIDIA Corporation.
- "Autodesk" and "Flame" are trademarks of Autodesk, Inc. (referenced
  only in the broader MCP ecosystem context, not in Vision3D itself).

---

## How to update this file

Add a new entry whenever you introduce a dependency that is not a standard
permissive license (MIT / BSD / Apache 2.0), or whenever you replace a
model component. Keep the "restrictive terms" section honest — if a user
downstream of Vision3D reads only one file, this is the one that prevents
a licensing mistake.
