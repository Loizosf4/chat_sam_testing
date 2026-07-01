# MoGe scene-graph spike

This directory is an isolated fixture and validation scaffold for a future MoGe scene-reconstruction vertical slice. It does **not** install MoGe or run inference, and it has no runtime dependency on the existing frontend, SAM backend, MCP server, or Blender tooling.

## Fixture

`inputs/office_test/` is copied from the repository's finalized SAM export for image `24457ea245d9417484c8bc2a235fea3c`:

- Source image: `data/images/24457ea245d9417484c8bc2a235fea3c.jpg`
- Final masks and semantic metadata: `data/exports/auto_scene_final_24457ea2/`
- Export metadata does not contain mask revisions, so every `mask_revision` is `null`.

The fixture files are immutable test inputs. Do not edit the image or masks in place. Their SHA-256 digests are recorded in `manifest.json`, and the copied files are marked read-only in this working tree.

No Blender bridge documentation was found in the repository during setup. The root `README.md` documents the SAM/MCP mask-export workflow only.

## Validate the fixture

From this experiment directory:

```powershell
python -m pip install -r requirements.txt
python -m pytest
python -m src.validate_fixture inputs/office_test/manifest.json
```

The validator requires each mask to be a PNG with one channel (or a palette whose resolved values remain binary), exactly two pixel values (`0` and `255`), the same dimensions as the source image, and at least one foreground pixel. It also checks unique object IDs, manifest structure, file hashes, and declared dimensions.

## Replace the fixture

1. Create a new directory under `inputs/`; do not overwrite `office_test`.
2. Copy one source image and a focused set of final, whole-object binary PNG masks into it. Copy from a finalized export, not a candidate or overlay directory.
3. Copy `office_test/manifest.json` as a starting point.
4. Assign a new `scene_id`; update the image filename, dimensions, source path, and SHA-256 digest.
5. For each mask, record a unique stable `object_id`, semantic label, local filename, repository-relative source export path, SHA-256 digest, and source revision. Use `null` only when the source has no revision metadata.
6. Keep representative coverage where available: large furniture and support surfaces, a chair/irregular object, a small object, a partially occluded object, and a wall-mounted object.
7. Mark the copied image and masks read-only and run both validation commands above.

Generated reconstruction artifacts belong in `outputs/`, which is ignored except for its placeholder.

## MoGe-2 inference environment

The official Microsoft documentation used for this spike is the MoGe repository README at commit `07444410f1e33f402353b99d6ccd26bd31e469e8`. The selected checkpoint is `Ruicheng/moge-2-vits-normal` at revision `679230677b4d282c6f304189a93e98e14f085902`: it provides metric points/depth and normals with 35M parameters, making it the appropriate official model for the detected 2 GiB NVIDIA MX250.

`environment.json` records the detected Python, PyTorch, CUDA, GPU, and model-selection details. `.venv` is created with system site packages so it can reuse the machine's CUDA-enabled PyTorch installation; MoGe and its Git dependencies are pinned in `requirements-moge.lock.txt`.

```powershell
.\setup_environment.ps1
.\.venv\Scripts\python.exe -m src.run_moge_inference
.\.venv\Scripts\python.exe -m src.smoke_test_output
```

The default command processes only `inputs/office_test/image.png`. It does not consume object masks. Numerical results are written to `outputs/office_test/moge/` as individual NPY files and a combined `geometry.npz`; JSON metadata and lossless PNG previews accompany them.

`image.png` is a lossless PNG encoding of the pixels in the original read-only `office_scene.jpg` fixture. Both remain read-only. The direct inference path intentionally omits MoGe's UI and mesh-export-only dependencies (`gradio`, `trimesh`, and `moderngl`).

## Geometry inspection

Generate previews, geometry-only and source-colored PLY point clouds, camera summaries, and JSON/Markdown quality reports from the persisted MoGe arrays:

```powershell
.\.venv\Scripts\python.exe -m src.generate_inspection_artifacts
```

Artifacts are written under `outputs/office_test/moge/inspection/`. The depth preview uses explicitly reported p2/p98 visualization clipping; raw NPY/NPZ values are never modified. Plane diagnostics use fixed image-space floor and wall regions, predicted-normal filtering, and deterministic RANSAC. They do not use the object masks.

## Per-object visible geometry

Extract deterministic geometry for the approved fixture masks without inferring hidden surfaces:

```powershell
.\.venv\Scripts\python.exe -m src.extract_object_geometry
.\.venv\Scripts\python.exe -m pytest -q
```

The main result is `outputs/office_test/geometry/object_geometry.json`. Each object has a diagnostic directory keyed by its stable object ID containing raw and conservatively filtered NPZ samples, previews, point clouds, and JSON/Markdown reports. Connected components remain supplementary parts of one semantic object.

Raw values use MoGe's OpenCV camera coordinates (+X right, +Y down, +Z forward) and retain its unverified metric scale. Scene-normalized values apply one reversible uniform transform based on the valid scene's p1/p99 bounds; raw values are never overwritten. Oriented boxes are PCA estimates of visible surfaces only and are omitted below the configured confidence/stability threshold.

## Deterministic scene evidence

The fixture additionally includes the finalized `desk.png` mask as one semantic object with three visible connected components. Build structural room planes, canonical coordinates, support patches, and relationship candidates with:

```powershell
.\.venv\Scripts\python.exe -m src.extract_object_geometry
.\.venv\Scripts\python.exe -m src.build_scene_evidence
.\.venv\Scripts\python.exe -m pytest -q
```

Outputs are `outputs/office_test/geometry/scene_evidence.json`, `scene_evidence.md`, and `scene_diagnostics/`. They are evidence records, not a final scene graph. Structural fitting excludes the six known object masks and uses normal clustering plus deterministic RANSAC. The canonical world is a reversible rigid transform: +Z follows the inward-oriented floor normal, +X is camera-right projected onto the floor, and +Y completes a right-handed basis. No camera extrinsics, hidden geometry, VLM conclusions, or Blender objects are introduced.
