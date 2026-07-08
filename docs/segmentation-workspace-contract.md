# Segmentation Workspace Contract

The segmentation workspace is a lightweight backend contract for the stage before
MoGe runs. It exists separately from the completed scene package because the full
scene package requires post-MoGe and reconstruction data such as camera
candidates, coordinate systems, structural room proxies, object transforms, and
generation status. This contract only records the source image and semantic
objects that will later receive finalized masks.

## Place in the Pipeline

The workspace starts after a source image is uploaded and before SAM prompting,
mask correction, MoGe execution, Blender sync, or scene-package export. New
workspaces begin in `draft` status and use schema version `1.0.0`.

## Stable Object Identity

Each semantic object receives a server-generated UUID in `object_id`. That ID is
stable when labels or display names change and is intended to become the final
mask ID, MoGe object ID, scene-package object ID, and Blender semantic ID. Object
identity is never derived from a label, filename, list position, or temporary SAM
mask identifier.

## Storage Layout

By default, workspaces are stored under `data/segmentation_workspaces/`. The root
can be overridden with `SEGMENTATION_WORKSPACE_ROOT`.

```text
data/segmentation_workspaces/{workspace_id}/
    workspace.json
    source/
        source.png | source.jpg | source.webp
    artifact-index.json
```

The artifact index maps application-controlled artifact IDs, such as
`source-images/{image_id}`, to managed relative paths, media types, and SHA-256
hashes. API responses expose artifact URLs, never filesystem paths.

## API Endpoints

```text
POST   /api/segmentation-workspaces
GET    /api/segmentation-workspaces/{workspace_id}
DELETE /api/segmentation-workspaces/{workspace_id}?expected_workspace_revision=...
POST   /api/segmentation-workspaces/{workspace_id}/objects
PATCH  /api/segmentation-workspaces/{workspace_id}/objects/{object_id}
DELETE /api/segmentation-workspaces/{workspace_id}/objects/{object_id}?expected_workspace_revision=...&expected_object_version=...
GET    /api/segmentation-artifacts/{kind}/{identifier}
```

## Optimistic Concurrency

Every successful object mutation increments `workspace_revision`. Object updates
increment that object's `object_version`; object creation starts at version `1`.
Mutations require the caller's expected current workspace revision, and object
updates or deletes also require the expected object version. Stale writes return
`409 Conflict`.

## Deferred Features

The following are intentionally outside this contract:

- SAM loading and predictions.
- Positive and negative prompt points.
- Candidate masks.
- Brush corrections.
- Final-mask composition.
- Quality reports.
- Export.
- MoGe execution.
- Frontend UI.
