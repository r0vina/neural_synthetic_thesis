## Repository Overview

| Script / File | Description |
|---|---|
| `background_principled.py` | Inverse rendering with a disney principled BSDF for the background, participating media for glass along with dielectric. |
| `bowl_advanced.py` | Diffuse BSDF for background, participating media and dielectric for glass. |
| `camera_converter.py` |Conversion from Colmap models exported as text files, to a mitsuba-compatible camera definition. |
| `convert_to_neus.py` | Conversion of Colmap cameras to the IDR-style format required by NeuS and AlphaNeuS |
| `exposure_inv.py` | Inverse rendering of the environment map, using L1 loss for each level of exposure (did not succeed). |
| `hdr_merge.py` | Attempted merging of the stepped captured RAW images into a single HDR. |
| `jerf_probe_pose_inverse.py` | Pose optimization for a reflective sphere to match the photos of the light probe from the ECO3D dataset |
| `lamp_envmap_pose.py` |Pose optimization for a reflective sphere, for the uncontrolled scene. |
| `lamp_envmap_inverse.py` | Inverse rendering environment map for the uncontrolled scene. |
| `lamp_texture_inverse.py` | Initial attempt at learning UV unwrapped texture, with a non-ideal scene. |
