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

## API Endpoints

```text
POST   /api/segmentation-workspaces
GET    /api/segmentation-workspaces/{workspace_id}
DELETE /api/segmentation-workspaces/{workspace_id}?expected_workspace_revision=...
POST   /api/segmentation-workspaces/{workspace_id}/objects
PATCH  /api/segmentation-workspaces/{workspace_id}/objects/{object_id}
DELETE /api/segmentation-workspaces/{workspace_id}/objects/{object_id}?expected_workspace_revision=...&expected_object_version=...
POST   /api/segmentation-workspaces/{workspace_id}/prepare-sam
POST   /api/segmentation-workspaces/{workspace_id}/objects/{object_id}/predict
POST   /api/segmentation-workspaces/{workspace_id}/objects/{object_id}/select-candidate
DELETE /api/segmentation-workspaces/{workspace_id}/objects/{object_id}/sam-draft?expected_workspace_revision=...&expected_object_version=...
GET    /api/segmentation-artifacts/{kind}/{identifier}
```

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

## Deferred Features

The following remain intentionally outside this contract:

- Frontend segmentation UI.
- Point-marker rendering.
- Latest-state-wins frontend request queue.
- Brush corrections.
- Manual override layers.
- Final-mask composition.
- Mask finalization.
- Quality reports.
- Export to the MoGe stage.
- MoGe pipeline integration.
- Blender integration.
