## Repository Overview

| Script / File | Description |
|---|---|
| `background_principled.py` | Inverse rendering with a disney principled BSDF for the background, participating media for glass along with dielectric. |
| `bowl_advanced.py` | Diffuse BSDF for background, participating media and dielectric for glass. |
| `camera_converter.py` |Conversion from Colmap models exported as text files, to a mitsuba-compatible camera definition. |
| `convert_to_neus.py` | Extracts image frames from captured videos using FFmpeg for use in COLMAP and downstream processing. |
| `align_pointclouds.py` | Applies rigid transformations between separate capture passes based on ICP or manually derived alignment parameters. |
| `build_mitsuba_scene.py` | Assembles Mitsuba scene dictionaries or XML scene descriptions from reconstructed geometry, materials, lighting, and camera parameters. |
| `optimize_envmap.py` | Runs inverse rendering to estimate an environment map from light-probe observations. |
| `optimize_materials.py` | Optimizes material parameters in Mitsuba against reference photographs using differentiable rendering. |
| `learn_table_texture.py` | Learns or optimizes a UV texture for the table/background surface from reference images. |
| `run_neusfacto.py` | Launches NeuSFacto / Nerfstudio experiments for geometry reconstruction of coated objects. |
| `convert_to_nerfstudio.py` | Converts image sets and COLMAP outputs into Nerfstudio-compatible dataset format. |
| `run_patchcore.py` | Runs the anomaly detection pipeline using DINO features and PatchCore / AnomalyDINO. |
| `render_synthetic_views.py` | Renders synthetic views from the reconstructed Mitsuba scene for qualitative comparison or anomaly detection experiments. |
| `requirements.txt` | Python dependencies required for the scripts in this repository. |
| `Dockerfile` | Container setup used to ensure compatibility for parts of the pipeline with strict CUDA or library requirements. |
