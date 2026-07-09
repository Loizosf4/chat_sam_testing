# Segmentation Workspace Contract

The segmentation workspace is a lightweight backend contract for the stage before
MoGe runs. It exists separately from the completed scene package because the full
scene package requires post-MoGe and reconstruction data such as camera
candidates, coordinate systems, structural room proxies, object transforms, and
generation status. This contract only records the source image and semantic
objects and draft SAM prompt state that will later produce finalized masks.

## Place in the Pipeline

The workspace starts after a source image is uploaded and before mask
finalization, MoGe execution, Blender sync, or scene-package export. New
workspaces begin in `draft` status and use schema version `1.0.0`.

## Stable Object Identity

Each semantic object receives a server-generated UUID in `object_id`. That ID is
stable when labels or display names change and is intended to become the final
mask ID, MoGe object ID, scene-package object ID, and Blender semantic ID. Object
identity is never derived from a label, filename, list position, or temporary SAM
mask identifier.

## SAM Draft State

Each segmentation object owns an independent `sam_draft`:

```text
sam_draft
    prompt_revision
    points
    box
    candidates
    selected_candidate_index
    prepared_image_key
    updated_at
```

Prompt points are source-image pixel coordinates:

```json
{ "x": 420, "y": 260, "label": 1 }
```

`label` is `1` for foreground/positive and `0` for background/negative. Point
coordinates must be finite and within the source-image bounds. Optional boxes use
`[x1, y1, x2, y2]` in source-image pixels.

`prompt_revision` orders interactive prediction requests independently from
normal `workspace_revision` and `object_version`. A prediction request must carry
a revision greater than the currently stored draft revision. Equal or older
prompt revisions return `409 Conflict`, including when a slower older prediction
finishes after a newer one has already been persisted.

Low-resolution SAM logits are kept in a bounded in-memory cache keyed by
workspace, object, prompt revision, and candidate index. They are not written to
`workspace.json`, exposed through HTTP, or expected to survive process restart.
When the referenced logits are missing, prediction falls back to the complete
current point and box prompt without `mask_input`.

Candidates are temporary draft mask artifacts. The highest-scoring candidate is
selected automatically after prediction unless a later selection request chooses
another candidate. `selected_candidate_index` references the current candidate
set; no semantic object ID changes when candidates are regenerated.

## Manual Mask Corrections

Each segmentation object also owns a `manual_mask` state for layered manual
corrections on top of the selected SAM candidate:

```text
manual_mask
    manual_revision
    base_prompt_revision
    base_candidate_index
    add_mask_url
    remove_mask_url
    composite_mask_url
    area_pixels
    bbox_xyxy
    updated_at
```

New objects start with `manual_revision` set to `0`, no base prompt or candidate
metadata, no artifact URLs, `area_pixels` set to `0`, `bbox_xyxy` set to
`[0, 0, 0, 0]`, and `updated_at` set to `null`.

When `manual_revision` is greater than zero, the state is anchored to the current
selected SAM candidate by `base_prompt_revision` and `base_candidate_index`.
Active manual states require all three artifact URLs and only expose
application-controlled URLs under
`/api/segmentation-artifacts/manual-masks/...`.

The persisted layers use this formula:

```text
composite_mask = (sam_base_mask OR manual_add_mask) AND NOT manual_remove_mask
```

The save endpoint accepts the complete edited binary mask that the user sees.
The backend derives:

```text
manual_add    = edited AND NOT base
manual_remove = base AND NOT edited
composite     = (base OR manual_add) AND NOT manual_remove
```

The composite must exactly match the uploaded edited mask. The add, remove, and
composite artifacts are full-resolution single-channel binary PNG files.

## Storage Layout

By default, workspaces are stored under `data/segmentation_workspaces/`. The root
can be overridden with `SEGMENTATION_WORKSPACE_ROOT`.

```text
data/segmentation_workspaces/{workspace_id}/
    workspace.json
    source/
        source.png | source.jpg | source.webp
    objects/
        {object_id}/
            sam/
                {prompt_revision}/
                    candidate-0.png
                    candidate-1.png
                    candidate-2.png
            manual/
                {manual_revision}/
                    add.png
                    remove.png
                    composite.png
    artifact-index.json
```

The artifact index maps application-controlled artifact IDs, such as
`source-images/{image_id}`, to managed relative paths, media types, and SHA-256
hashes. SAM candidate artifacts use keys such as
`sam-candidates/{workspace_id}/{object_id}/{prompt_revision}/{candidate_index}`.
API responses expose artifact URLs, never filesystem paths.

Only the latest candidate revision for an object is retained. After a newer
prediction is persisted, older candidate artifact-index entries and older
candidate artifact directories for that object are removed. Candidate masks are
full-resolution binary PNGs served with `Cache-Control: no-store` because they
are interactive drafts.

Manual artifacts use keys such as
`manual-masks/{workspace_id}/{object_id}/{manual_revision}/add`. Only the latest
manual revision is retained after a successful replacement. Manual artifact
responses are served with `Cache-Control: no-store` and
`X-Content-Type-Options: nosniff`.

## API Endpoints

```text
GET    /segment
POST   /api/segmentation-workspaces
GET    /api/segmentation-workspaces/{workspace_id}
DELETE /api/segmentation-workspaces/{workspace_id}?expected_workspace_revision=...
POST   /api/segmentation-workspaces/{workspace_id}/objects
PATCH  /api/segmentation-workspaces/{workspace_id}/objects/{object_id}
DELETE /api/segmentation-workspaces/{workspace_id}/objects/{object_id}?expected_workspace_revision=...&expected_object_version=...
POST   /api/segmentation-workspaces/{workspace_id}/prepare-sam
POST   /api/segmentation-workspaces/{workspace_id}/objects/{object_id}/predict
POST   /api/segmentation-workspaces/{workspace_id}/objects/{object_id}/select-candidate
PUT    /api/segmentation-workspaces/{workspace_id}/objects/{object_id}/manual-mask
DELETE /api/segmentation-workspaces/{workspace_id}/objects/{object_id}/manual-mask?expected_workspace_revision=...&expected_object_version=...&expected_manual_revision=...
DELETE /api/segmentation-workspaces/{workspace_id}/objects/{object_id}/sam-draft?expected_workspace_revision=...&expected_object_version=...
GET    /api/segmentation-artifacts/{kind}/{identifier}
```

Manual mask save uses `multipart/form-data` with `edited_mask`,
`base_prompt_revision`, `base_candidate_index`, `expected_workspace_revision`,
`expected_object_version`, and `expected_manual_revision`. `edited_mask` must be
a grayscale binary PNG with the exact source-image dimensions. The backend does
not accept a client-provided base mask path or base mask file.

## Browser Workspace

`/segment` is the first frontend for this pre-MoGe contract. It supports image
upload, loading an existing workspace with `/segment?workspace={workspace_id}`,
semantic object creation, object rename/delete, SAM preparation, positive and
negative point tools, candidate-mask display, candidate selection, undo, draft
reset, and pan/zoom.

The browser keeps local prompt state separate from the persisted `sam_draft`.
Point markers are added to local cumulative state immediately so the user sees
feedback before SAM returns. Each local prompt change consumes a new
monotonically increasing prompt revision; failed or stale revisions are not
reused.

The frontend uses a latest-state-wins prediction controller. If a point is added
while a prediction is running, only the newest cumulative state is submitted when
the active request finishes. Older responses cannot replace a newer applied
mask, and stale `409` prediction responses are ignored when a newer local
revision exists.

Candidate selection is a structural mutation and is blocked while the selected
object has an active or pending prediction. Candidate masks are displayed from
their artifact URLs with a cache-busting query string; stored workspace URLs are
not modified.

## Optimistic Concurrency

Every successful object mutation increments `workspace_revision`. Object updates
increment that object's `object_version`; object creation starts at version `1`.
Mutations require the caller's expected current workspace revision, and object
updates or deletes also require the expected object version. Stale writes return
`409 Conflict`.

Interactive prediction is different: rapid prompt changes are ordered by
`prompt_revision`, not by caller-supplied workspace/object versions. A successful
prediction still increments `workspace_revision` and the object's
`object_version`, but the prediction endpoint does not require those expected
versions. Candidate selection and draft clearing are normal explicit mutations,
so they require expected workspace and object versions.

Prediction follows a latest-state-wins-compatible pattern: validate the prompt
revision, run SAM without holding the workspace file lock for the full inference,
then reacquire the lock and reject the result if an equal or newer prompt revision
has already been stored.

Manual mask save and clear require workspace, object, and manual optimistic
concurrency values. A successful manual save increments `manual_revision`,
`object_version`, and `workspace_revision`. A successful clear resets the manual
state, increments object/workspace revisions, and preserves the SAM prompt,
candidate list, and selected candidate.

Saved manual corrections are anchored to the selected SAM candidate. While a
manual revision is active, the backend rejects new SAM prediction persistence and
selection of a different candidate with `409 Conflict`. Clearing the SAM draft is
an explicit destructive reset and clears manual corrections and their artifacts
in the same successful mutation. Object deletion removes manual artifact-index
entries and the object artifact directory.

## Deferred Features

The following remain intentionally outside this contract:

- Brush corrections.
- Frontend brush UI.
- Candidate-mask quality reports.
- Final-mask finalization.
- Mask finalization.
- Quality reports.
- Export to the MoGe stage.
- MoGe pipeline integration.
- Blender integration.
