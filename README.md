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
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Open `frontend/index.html` directly in your browser, or open http://127.0.0.1:8000.

## Local Files

The virtual environment, Python bytecode, `.env`, and uploaded runtime data are ignored by git. Recreate the virtual environment locally instead of committing `.venv/` or `__pycache__/`.

## Phase 1

The app can upload an image to `data/images/`, return its generated image ID and dimensions, and display it on the frontend canvas. SAM, points, boxes, masks, and MCP are not implemented yet.
