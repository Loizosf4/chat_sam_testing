# Local SAM Mask Editor

A local-first object mask editor scaffold.

## Architecture

- Python FastAPI backend
- Plain HTML, CSS, and JavaScript frontend
- Local SAM model integration in a later phase
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

## Phase 1

The app can upload an image to `data/images/`, return its generated image ID and dimensions, and display it on the frontend canvas. SAM, points, boxes, masks, and MCP are not implemented yet.
