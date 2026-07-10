# Local SAM Mask Editor

A local-first object mask editor scaffold.

## Architecture

- Python FastAPI backend
- Plain HTML, CSS, and JavaScript frontend
- Local Meta Segment Anything Model 1 integration through `segment_anything`
- No React
- No database
- No cloud APIs

## File Structure

```text
backend/
  main.py
  sam_engine.py
  mask_ops.py
frontend/
  index.html
  app.js
  style.css
data/
  images/
  masks/
  exports/
requirements.txt
.env.example
README.md
```

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

The runner uses http://127.0.0.1:8000 when available. If port `8000` is occupied, it automatically tries the next free port through `8010` and prints the URL.

Open the printed URL, or open `frontend/index.html` directly in your browser. When opened directly, the frontend probes local ports `8000` through `8010` to find the backend.

## Local SAM 1 Configuration

This project uses Meta's original SAM 1 Python package, `segment_anything`, through `backend/sam_engine.py`. It expects a SAM 1 checkpoint whose architecture matches `SAM_MODEL_TYPE`.

For the local CPU ViT-B setup, create `.env` from `.env.example` or set these variables in PowerShell:

```powershell
$env:SAM_CHECKPOINT = 'I:\Models\SAM1_CPU\segment-anything\checkpoints\sam_vit_b_01ec64.pth'
$env:SAM_MODEL_TYPE = 'vit_b'
$env:SAM_DEVICE = 'cpu'
```

`SAM_CHECKPOINT_PATH` is still accepted as a backward-compatible alias, but `SAM_CHECKPOINT` is preferred.

Do not commit `.env`; it is ignored by git. The checkpoint remains outside the repository.

The repository `.venv` contains the app dependencies, but PyTorch is not pinned in `requirements.txt` because CPU and GPU builds are installed from different package indexes. For this machine, the existing SAM environment has CPU PyTorch and SAM installed, but it does not currently include the FastAPI/MCP app dependencies. The safer option is to use the project `.venv` for the app and install a CPU PyTorch build into it, or install the app requirements into the SAM environment after checking versions. Do not reinstall or upgrade PyTorch, TorchVision, NumPy, or OpenCV without reviewing the current versions first.

Validate the configured checkpoint and load the model without processing a dataset:

```powershell
.\.venv\Scripts\python.exe scripts\validate_sam_model.py
```

If you choose to run from the existing SAM environment instead, first ensure it contains the app dependencies from `requirements.txt`, then start the app with:

```powershell
I:\Models\SAM1_CPU\.venv\Scripts\Activate.ps1
python run.py
```

## Local Files

The virtual environment, Python bytecode, `.env`, and uploaded runtime data are ignored by git. Recreate the virtual environment locally instead of committing `.venv/` or `__pycache__/`.

## MCP Server

Install requirements, then run the MCP server over stdio:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m backend.mcp_server
```

The MCP server exposes these tools:

- `sam_health`
- `sam_load_model`
- `sam_register_image`
- `sam_set_image`
- `sam_predict`
- `sam_merge_masks`
- `sam_subtract_masks`
- `sam_refine_mask`
- `sam_export_masks`
- `sam_preview_mask_overlay`
- `sam_preview_masks_overlay`
- `sam_preview_candidate_contact_sheet`
- `sam_mask_quality_report`

`sam_health` is safe to call before loading SAM. `sam_load_model` uses `.env` by default, or accepts optional `checkpoint_path`, `model_type`, and `device` arguments. `sam_register_image` accepts a local file path only and copies the image into `data/images/`.

The prediction and mask operation MCP tools call the same Python modules used by the FastAPI app. They return local file paths for generated masks and exports rather than base64 preview strings.

## Recommended MCP Masking Workflow

Use this workflow when an agent needs high-quality object masks:

1. Register the source image with `sam_register_image`.
2. Check status with `sam_health`, then load the model with `sam_load_model`.
3. Identify the requested semantic objects before predicting. One semantic object should become one final mask.
4. For each object, call `sam_set_image` as needed and generate multiple candidates with `sam_predict`.
5. Use positive points on the object and negative points on nearby objects or background. Avoid room structure, walls, floors, counters, shadows, and other structural surfaces unless the task explicitly asks for them.
6. Use `sam_preview_candidate_contact_sheet` to compare candidates for the same object.
7. Inspect colored overlays with `sam_preview_mask_overlay` or `sam_preview_masks_overlay`; do not judge quality from black/white masks alone.
8. Choose or refine candidates with `sam_refine_mask`, `sam_merge_masks`, and `sam_subtract_masks`.
9. Merge temporary component masks only when they belong to one semantic object. Split separate real objects even when they touch, share color, or look similar; do not merge separate cabinets, chairs, desks, shelves, or furniture pieces unless the user asked for a grouped region.
10. Run `sam_mask_quality_report` on the proposed final masks and review warnings for overlap, tiny masks, many components, border-touching masks, and suspicious bbox matches.
11. Export final whole-object masks with `sam_export_masks` and `include_previews: true`.
12. Inspect `previews/all_masks_overlay.png` before reporting that the mask set is done.

Temporary component masks are fine during construction, but only final whole-object masks should be exported. Always report uncertain masks and explain what should be checked visually.

## Local MoGe 2 Configuration

MoGe runs out-of-process through the configured MoGe Python environment. The
FastAPI/SAM process does not import `torch` or `moge`, so the Python 3.10 CPU
SAM environment stays separate from the Python 3.12 AMD ROCm MoGe environment.

Create or update your untracked `.env` with local values:

```powershell
$env:MOGE_PYTHON = '<path-to-moge-venv-python.exe>'
$env:MOGE_REPOSITORY = '<path-to-local-moge-repository>'
$env:MOGE_CHECKPOINT = '<path-to-moge-2-vitl-normal-model.pt>'
$env:MOGE_VERSION = 'v2'
$env:MOGE_DEVICE = 'cuda'
$env:MOGE_MODEL = 'moge-2-vitl-normal'
```

Do not set `MOGE_DEVICE=rocm` or `hip`; ROCm PyTorch exposes the AMD GPU
through the CUDA-facing API, so the device string remains `cuda` or `cuda:0`.
CPU execution is disabled unless `MOGE_ALLOW_CPU=true` is set explicitly.

Validate the configured worker, editable MoGe import, AMD compatibility fix,
PyTorch/ROCm stack, GPU visibility, and checkpoint load:

```powershell
.\.venv\Scripts\python.exe scripts\validate_moge_model.py
```

The MoGe worker receives a registered RGB source image path and writes numerical
geometry under `data/exports/moge/{image_id}/` by default:

- `geometry.npz`
- `points.npy`
- `depth.npy`
- `normal.npy`
- `valid_mask.npy`
- `intrinsics.npy`
- preview PNGs
- `metadata.json`

The existing reconstruction pipeline consumes MoGe points, depth, normals,
validity mask, and normalized intrinsics. It does not require GLB or PLY output
for the checked-in scene-package flow. SAM masks and MoGe geometry are produced
independently from the same source image; downstream reconstruction filters the
full-image MoGe maps with the final SAM object masks.

## Managed Segmentation Reconstruction Jobs

Pre-MoGe segmentation exports can now start persistent managed reconstruction
jobs through the backend API. A job is anchored to one immutable segmentation
export and the workspace-managed source image, runs MoGe through the isolated
worker, then invokes the generic Unified V3 compiler with `RECONSTRUCTION_PYTHON`.
Jobs are stored under the owning workspace in `reconstruction-jobs/`, and
successful immutable outputs are published under `reconstructions/{job_id}/`.

Required local settings:

```powershell
$env:RECONSTRUCTION_PYTHON = '<path-to-reconstruction-venv-python.exe>'
$env:RECONSTRUCTION_COMPILER_TIMEOUT_SECONDS = '1800'
$env:RECONSTRUCTION_MAX_QUEUED_JOBS = '8'
```

The reconstruction Python must include `numpy`, `Pillow`, `scipy`, `pydantic`,
and `jsonschema`. Job status is available through:

```text
POST /api/segmentation-workspaces/{workspace_id}/exports/{export_id}/reconstructions
GET  /api/segmentation-workspaces/{workspace_id}/reconstructions
GET  /api/segmentation-workspaces/{workspace_id}/reconstructions/{job_id}
GET  /api/segmentation-reconstruction/health
```

The backend processes one reconstruction at a time. Queued or running jobs are
marked `interrupted` after restart; retry means creating a new job. Active jobs
block workspace deletion, but normal object edits and new exports remain
allowed.

Successful completed jobs can be explicitly added to scene review from the
selected job details in `/segment`. The bridge is server-side and trusted: it
uses the exact immutable export, exact Unified scene plan, managed source image,
and managed reconstruction artifacts already stored by the backend. The browser
does not upload, download, or re-upload the export ZIP, scene plan, MoGe
geometry, or source image, and it does not call the generic multipart
`/api/scenes/import` endpoint.

```text
GET  /api/segmentation-workspaces/{workspace_id}/reconstructions/{job_id}/review-scene
POST /api/segmentation-workspaces/{workspace_id}/reconstructions/{job_id}/review-scene
```

The POST requires the current job version and is idempotent. A first import
returns a self-contained scene package and later calls return the existing scene
when provenance matches. Jobs with compiler quality gates passed can be imported
directly. Jobs with `compilation_passed=false` are still successful, but require
an explicit review-required acknowledgement before import.

Imported scenes preserve stable object IDs and reconstruction provenance, copy
the source image, SAM masks/overlays, Unified scene plan, reconstruction result
manifest, compilation reports, MoGe geometry, and sanitized MoGe summary into
the scene-package store, and open in the existing review UI with `/?scene=...`.
They remain available after the segmentation workspace is deleted. The review UI
shows a `Back to segmentation workspace` link for scenes created from
reconstruction-job provenance. Blender execution and asset replacement remain
separate, deferred steps.
