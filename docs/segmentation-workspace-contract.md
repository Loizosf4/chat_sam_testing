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

## Immutable Export Snapshots

A segmentation export is an immutable, revisioned snapshot of the currently
persisted effective masks. The workspace remains editable after export; later
SAM prompts, candidate changes, manual saves, renames, or object deletion do not
modify or delete older export directories, artifact-index entries, hashes, or
records. A user can create a newer export after making corrections.

For each object, the exported effective mask is selected in this order:

```text
1. Saved manual composite, when manual_revision > 0
2. Selected SAM candidate
```

Unsaved browser brush pixels are not visible to the backend. The future export
UI must save local brush edits before requesting an export.

Each export record stores:

```text
export_id
created_at
created_from_workspace_revision
published_workspace_revision
masks
metadata_url
quality_report_url
quality_markdown_url
combined_preview_url
archive_url
archive_sha256
warning_count
mask_count
total_mask_area
```

Each exported mask snapshot stores the stable `object_id`, current object
version, label/display name, effective source kind and anchors, exported mask
filename, mask and preview URLs, SHA-256, area, and bbox. The stable
segmentation `object_id` is written byte-for-byte as the final SAM `mask_id` in
`metadata.json`.

`created_from_workspace_revision` is the workspace revision that was snapshotted.
`published_workspace_revision` is the revision after appending the export record,
so a successful export increments `workspace_revision` exactly once without
incrementing object versions or changing SAM/manual state.

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
    exports/
        {export_id}/
            metadata.json
            mask_quality_report.json
            mask_quality_report.md
            {safe_mask_filename}.png
            previews/
                {safe_mask_filename_stem}_overlay.png
                all_masks_overlay.png
            segmentation-export.zip
    reconstruction-jobs/
        {job_id}.json
    reconstructions/
        {job_id}/
            moge/
                geometry.npz
                moge-summary.json
                depth_preview.png
                normal_preview.png
                valid_mask_preview.png
            scene/
                unified_scene_plan.json
                compilation_report.json
                compilation_report.md
                room_plan.json
                camera_candidates.json
                object_pose_report.json
                placement_report.json
                collision_report.json
                confidence_report.json
                support_graph.json
                clean_scene_plan_overview.png
            handoff/
                blender_one_batch_manifest.json
            reconstruction-result-manifest.json
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

Export artifacts use immutable keys such as:

```text
workspace-exports/{workspace_id}/{export_id}/archive
workspace-exports/{workspace_id}/{export_id}/metadata
workspace-exports/{workspace_id}/{export_id}/quality-json
workspace-exports/{workspace_id}/{export_id}/quality-markdown
workspace-exports/{workspace_id}/{export_id}/combined-preview
workspace-exports/{workspace_id}/{export_id}/masks/{object_id}
workspace-exports/{workspace_id}/{export_id}/previews/{object_id}
```

All export artifact paths are managed relative paths under the owning workspace,
hash-verified on retrieval, and served through
`/api/segmentation-artifacts/workspace-exports/...` with
`X-Content-Type-Options: nosniff`. The ZIP media type is `application/zip`;
Markdown is served as text/Markdown.

`segmentation-export.zip` contains only safe relative members:

```text
metadata.json
mask_quality_report.json
mask_quality_report.md
final mask PNG files
previews/
```

It never contains itself, workspace JSON, the artifact index, temporary files, or
filesystem paths.

Reconstruction result artifacts use immutable keys such as:

```text
reconstruction-results/{workspace_id}/{job_id}/result-manifest
reconstruction-results/{workspace_id}/{job_id}/scene-plan
reconstruction-results/{workspace_id}/{job_id}/compilation-report
reconstruction-results/{workspace_id}/{job_id}/compilation-markdown
reconstruction-results/{workspace_id}/{job_id}/room-plan
reconstruction-results/{workspace_id}/{job_id}/camera-candidates
reconstruction-results/{workspace_id}/{job_id}/object-pose-report
reconstruction-results/{workspace_id}/{job_id}/placement-report
reconstruction-results/{workspace_id}/{job_id}/collision-report
reconstruction-results/{workspace_id}/{job_id}/confidence-report
reconstruction-results/{workspace_id}/{job_id}/support-graph
reconstruction-results/{workspace_id}/{job_id}/blender-manifest
reconstruction-results/{workspace_id}/{job_id}/moge-geometry
reconstruction-results/{workspace_id}/{job_id}/moge-summary
```

Optional reconstruction preview artifacts include `overview`,
`projected-primitives`, `room-camera`, `confidence-overview`,
`ambiguity-overview`, `depth-preview`, `normal-preview`, and
`valid-mask-preview`. Raw MoGe `metadata.json` and compiler clean-input audits
contain local paths and are not registered as browser artifacts.

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
POST   /api/segmentation-workspaces/{workspace_id}/exports
GET    /api/segmentation-workspaces/{workspace_id}/exports
GET    /api/segmentation-workspaces/{workspace_id}/exports/{export_id}
POST   /api/segmentation-workspaces/{workspace_id}/exports/{export_id}/reconstructions
GET    /api/segmentation-workspaces/{workspace_id}/reconstructions
GET    /api/segmentation-workspaces/{workspace_id}/reconstructions/{job_id}
GET    /api/segmentation-workspaces/{workspace_id}/reconstructions/{job_id}/review-scene
POST   /api/segmentation-workspaces/{workspace_id}/reconstructions/{job_id}/review-scene
GET    /api/segmentation-reconstruction/health
GET    /api/segmentation-artifacts/{kind}/{identifier}
```

Manual mask save uses `multipart/form-data` with `edited_mask`,
`base_prompt_revision`, `base_candidate_index`, `expected_workspace_revision`,
`expected_object_version`, and `expected_manual_revision`. `edited_mask` must be
a grayscale binary PNG with the exact source-image dimensions. The backend does
not accept a client-provided base mask path or base mask file.

Export creation uses `application/json`:

```json
{
  "expected_workspace_revision": 12,
  "expected_objects": [
    {"object_id": "stable-object-id", "expected_object_version": 5}
  ],
  "include_previews": true
}
```

`expected_objects` must contain the exact current object set, with no missing,
unknown, or duplicate IDs, and every object version must match. Stale workspace
or object versions return `409 Conflict`. A valid export requires at least one
object and one non-empty binary effective mask per object.

Export GET responses include computed `is_stale`. The persisted immutable export
record is not modified when staleness changes.

Reconstruction creation uses `application/json`:

```json
{
  "expected_workspace_revision": 18,
  "expected_export_archive_sha256": "64-character-sha256",
  "resolution_level": 9,
  "num_tokens": null
}
```

The response is `202 Accepted` with a queued reconstruction job. The backend
does not infer "latest export" when an export ID is supplied. Revision or export
archive hash conflicts return `409 Conflict`; missing workspace/export/job
resources return `404`.

## Browser Workspace

`/segment` is the first frontend for this pre-MoGe contract. It supports image
upload, loading an existing workspace with `/segment?workspace={workspace_id}`,
semantic object creation, object rename/delete, SAM preparation, positive and
negative point tools, candidate-mask display, candidate selection, undo, draft
reset, manual add/remove brush corrections, and pan/zoom.

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

Manual brush tools in `/segment` edit a local complete binary composite mask in
source-image pixels. The displayed selected-object mask uses this priority:

```text
1. Current local brush editor mask
2. Saved manual composite mask
3. Selected SAM candidate mask
4. No mask
```

Local brush edits are not persisted until the user saves corrections through the
manual-mask endpoint. Saving sends a full-resolution binary PNG of the edited
mask, the current base SAM prompt revision and candidate index, and expected
workspace/object/manual revisions. A successful save rebases the local editor to
the saved composite, clears local undo/redo history, and preserves SAM points and
candidates.

Clearing saved manual corrections calls the manual clear endpoint, removes the
persisted manual state, and reloads the selected SAM candidate as the brush base.
It does not clear SAM points or candidates. Resetting the SAM draft is separate:
that destructive action clears SAM points, SAM candidates, saved manual
corrections, and unsaved local brush edits.

While saved manual corrections exist, `/segment` disables positive/negative SAM
point prompting and different-candidate selection until the manual corrections
are cleared. While unsaved local brush edits exist, `/segment` also blocks SAM
base changes and warns before object/workspace switches, object deletion, SAM
draft reset, or browser navigation would discard those local edits.

`/segment` also exposes an Export snapshots panel backed by the immutable export
API. The panel shows export readiness, current/stale export history, artifact
links, combined preview, per-mask snapshot metadata, and a lazily loaded quality
report. Export creation is blocked until every object has a persisted effective
mask and there are no unsaved brush edits, manual save/clear operations, active
SAM predictions, object mutations, or another export in progress. While an
export request is active, workspace loading/upload, object CRUD, SAM prompting,
candidate selection, SAM draft reset, manual painting, brush undo/redo/reset,
manual save/clear, and a second export request are disabled. Pan, zoom, inspect,
and overlay opacity remain local view controls and stay available.

The browser captures the workspace revision and exact object/version set before
calling the export endpoint. A stale `409 Conflict` refreshes the active
workspace and export history without retrying automatically. Export-history,
export-create, and quality-report responses are applied only when they still
belong to the currently active workspace and selected export.

`/segment` also exposes a Reconstruction panel for managed reconstruction jobs
created from immutable export snapshots. The panel checks
`GET /api/segmentation-reconstruction/health` when a workspace is activated and
when the user manually refreshes reconstruction status. It displays only
browser-safe health fields returned by the API: configured state, MoGe/compiler
availability, worker running state, global queued/running counts, device, model,
and controlled error text.

Starting reconstruction requires an active workspace, a selected export,
loaded configured health, a valid current workspace revision, a selected export
archive SHA-256, no export creation in progress, no other start submission in
flight, a resolution level from `1` through `9`, and either an empty token
override or a positive integer. Empty token override is sent as `null`. A stale
selected export is allowed because the job uses the export's immutable saved
masks; the UI warns that reconstruction will use those saved masks instead of
the current editable workspace.

The browser captures an immutable start-operation snapshot before posting:
workspace ID, workspace revision, export ID, export archive SHA-256, resolution
level, token override, operation ID, and reconstruction workspace generation.
Later workspace, export-selection, settings changes, health refreshes,
job-history refreshes, or polling updates do not mutate or invalidate that
request. A workspace switch or reconstruction state reset does invalidate it.
`409 Conflict` does not auto-retry; the UI refreshes the workspace, export
history, and reconstruction history and preserves useful backend wording. `503
Service Unavailable` refreshes health and job history because a worker
submission failure may still have persisted a terminal failed job. Network or
other failures clear only the start busy state and do not create fake local
jobs. Pressing Enter in the reconstruction settings form prevents normal page
submission and invokes the same validated start action as the button.

Reconstruction job history is loaded from
`GET /api/segmentation-workspaces/{workspace_id}/reconstructions` and polled
with recursive timeout polling while the active workspace has queued or running
jobs. The browser uses backend `progress_percent` as-is and renders the stages
as waiting, validating, MoGe, compiling, publishing, complete, failed, or
interrupted. Polling is invalidated on workspace switch, avoids overlapping list
requests, uses bounded backoff on failures, and refreshes immediately when the
document becomes visible. Late responses are guarded by workspace and generation
identity and cannot populate another workspace.

The job list is sorted newest first and merged by `job_version`: newer records
win, older polled records cannot replace newer terminal state, and a successful
list response is authoritative for persisted jobs in that workspace. Every job
is anchored to `export_id` and `export_archive_sha256`; selecting an export
shows its related jobs, while the workspace-wide history keeps all persisted
jobs including retries. If a job references an export no longer present in
export history, the UI displays that the referenced export is unavailable and
disables retry.

The selected-job summary shows job ID/version, status, stage, progress,
timestamps, export ID, export archive SHA, stale-at-start state, scene ID,
semantic object count, resolution level, and token override. `status=succeeded`
with `result.compilation_passed=true` is rendered as reconstruction succeeded
and quality gates passed. `status=succeeded` with
`result.compilation_passed=false` remains a successful job and is rendered as
review recommended, not as a failed job. Failed and interrupted jobs show the
controlled error code, message, stage, and retryable flag without stack traces.

Retry is explicit and creates a new reconstruction job. It is available only for
failed or interrupted retryable jobs whose export still exists, when health is
configured and no start request is active. The retry uses the current workspace
revision, the original job's export ID, the current immutable export record's
archive SHA-256, and the previous job's resolution and token settings. The old
job remains in history.

Succeeded jobs expose browser-safe result artifacts through the exact backend
URLs returned in the job record. The UI links to the result manifest, Unified
scene plan, compilation reports, room plan, camera candidates, pose, placement,
collision, confidence, support graph, Blender-neutral manifest, and sanitized
MoGe summary, and uses a download link for MoGe geometry without fetching
`geometry.npz` into JavaScript. Optional previews are shown in a responsive grid
when present and skipped cleanly when absent.

Compilation report JSON and sanitized MoGe summary JSON are loaded lazily when
their details sections are opened, cached by workspace, job, and artifact URL,
and guarded against late responses. Compilation and MoGe diagnostics use
independent per-artifact request generations, so they may load concurrently
without invalidating each other. A diagnostic may finish and cache while another
job is selected; returning to the original job displays the cached result. The
compilation viewer renders dynamic quality gates without hardcoding a fixed
gate set and treats false gates as diagnostics. The MoGe summary viewer renders
returned sanitized primitive browser-safe fields and defensively hides strings
that resemble absolute paths such as `C:\`, `/...`, or `file://...`.

Queued or running reconstruction jobs do not freeze the editable segmentation
workspace. Workspace loading/upload, object creation/rename/delete, SAM prompts,
candidate selection, SAM draft reset, manual add/remove painting, brush
undo/redo/reset, manual save/clear, and export creation retain their existing
guards. The reconstruction frontend prevents only a second concurrent start
submission, starting while export creation is pending, starting without an
archive hash, or starting while health is unavailable. Switching workspaces
invalidates frontend polling and clears old reconstruction UI state but sends no
backend cancellation request; returning to the old workspace reloads its
persisted job history.

Successful completed reconstruction jobs can be explicitly added to the existing
scene-review UI through the selected job's `Scene review` action. This is a
trusted server-side bridge, not a browser upload: the browser sends only the
workspace ID, job ID, expected job version, and, when needed, a review-required
acknowledgement. It never downloads and re-uploads the export ZIP, Unified scene
plan, MoGe geometry, or source image, and it does not call the generic multipart
`POST /api/scenes/import` endpoint.

The bridge accepts only jobs with `status=succeeded`, `stage=complete`, a
present result, and a scene ID. The current `job_version` must match
`expected_job_version`; terminal jobs are not mutated and workspace revision is
not incremented. A successful job whose `result.compilation_passed=false` is
eligible, but the POST requires `acknowledge_review_required=true`. The UI
distinguishes quality gates passed from review required and uses a visible
acknowledgement before creating review scenes from review-required results.

Before import, the backend revalidates the exact immutable export archive SHA,
export metadata, final masks, overlays, workspace-managed source image hash and
dimensions, reconstruction result manifest, Unified scene plan, compilation
reports, MoGe geometry, and sanitized MoGe summary through managed artifact
indexes. Object IDs must match the reconstruction job exactly and duplicates are
rejected. No browser-supplied paths are accepted.

Review scene creation is idempotent per reconstruction job. The first POST
returns `201` with `created=true`; later POSTs for the same job return `200`
with `created=false` after verifying scene provenance. A scene-ID collision with
unrelated provenance returns `409` and is never overwritten. The GET status
endpoint derives import state from the scene store and provenance so page
reloads rediscover existing review scenes.

The imported scene package is self-contained. The scene store copies the source
image, SAM metadata, masks, overlays, Unified scene plan, optional mask quality
report, reconstruction result manifest, compilation report JSON, compilation
Markdown, `moge/geometry.npz`, and sanitized `moge-summary.json`. It records
browser-safe scene provenance in `Scene.metadata` and exposes `/artifacts/`
scene-input URLs. The scene remains loadable through `/?scene={scene_id}` after
the segmentation workspace is deleted. The review UI shows a
`Back to segmentation workspace` link only for scenes with reconstruction-job
provenance.

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

Export publication is transactional under the workspace lock. The backend
validates optimistic concurrency, resolves and hash-verifies effective masks,
writes metadata, mask PNGs, quality reports, previews, and ZIP into a staging
directory, then atomically publishes the export directory, artifact-index entries,
and workspace record. If publication fails, the workspace revision and artifact
index are restored and partial export directories are removed.

An export is stale when the current workspace no longer has the same object IDs,
object versions, labels, effective source kind, SAM prompt revision, selected
candidate index, manual revision, or effective mask SHA-256. Adding a newer
export record alone does not make an older export stale.

## Export Compatibility

Export `metadata.json` follows the existing final-SAM adapter format:

```text
image_id
original_filename
width
height
exported_at
masks[]
```

Each mask entry includes `label`, `mask_id`, `filename`, `area`, and `bbox`;
preview-enabled entries also include `color` and `preview_path`. `mask_id` is
the stable workspace `object_id`, allowing `backend.scene_package.adapter` to
map it directly to the future scene-package `object_id`. The backend derives
area and bbox from exported pixels and never trusts stale client metadata.

The export quality report is written as both JSON and Markdown and includes
per-mask metrics, connected components, border-touch diagnostics, pairwise
overlaps, bbox comparisons, and warnings. Warnings do not fail export creation.

## Generic Reconstruction Compiler Boundary

The experiment Unified V3 clean compiler is ready to be invoked by a future
managed reconstruction job. It accepts the immutable export directory, source
image, persisted MoGe directory, a new output directory, stable scene ID, and an
optional new Blender-neutral handoff directory. It supports variable object
counts and arbitrary or duplicate semantic labels. Export `mask_id` values are
preserved exactly as reconstructed `object_id` values.

The compiler validates source dimensions, safe mask filenames, unique IDs,
non-empty binary masks, required MoGe arrays and shapes, finite usable geometry,
and metadata dimensions before creating staging output. It reads only those
inputs and checked-in generic schemas. Output and optional handoff publication
are atomic and do not overwrite existing directories.

The current algorithm remains scoped to indoor reconstruction and requires
evidence for a floor and two room walls (`floor`, `left_wall`, and `right_wall`).
Insufficient structural evidence is a hard compiler error. Generic quality gates
cover identity, primitive and transform validity, supports, normal-first
completion, clean reads, room/camera output, placements, and collisions.
Review-quality gate failure is recorded as `passed=false` and is distinct from a
hard input or contract failure.

The exact repository-root invocation is:

```powershell
& $env:RECONSTRUCTION_PYTHON -m experiments.moge_scene_graph_spike.src.compile_unified_v3_scene `
  --mode clean_reconstruction `
  --sam-dir <immutable-export-directory> `
  --source-image <source-image-path> `
  --moge-dir <moge-output-directory> `
  --output-dir <new-reconstruction-output-directory> `
  --scene-id <stable-scene-id> `
  --handoff-dir <optional-new-handoff-directory>
```

`RECONSTRUCTION_PYTHON` must provide `numpy`, `Pillow`, `scipy`, `pydantic`, and
`jsonschema`; it is deliberately separate from `MOGE_PYTHON`.
`RECONSTRUCTION_COMPILER_TIMEOUT_SECONDS` defaults to `1800`, and
`RECONSTRUCTION_MAX_QUEUED_JOBS` defaults to `8`.

## Managed Reconstruction Jobs

A reconstruction job is persisted separately from the segmentation workspace
under `reconstruction-jobs/{job_id}.json`. Job state never increments
`workspace_revision`, never mutates export records, and never increments object
versions. Workspace editing remains allowed while a job runs. The job input is
anchored to the selected immutable export directory and managed source image.

Job statuses are `queued`, `running`, `succeeded`, `failed`, and `interrupted`.
Job stages are `queued`, `validating`, `moge`, `compiling`, `publishing`,
`complete`, `failed`, and `interrupted`. `job_version` starts at `1` and
increments for each persisted state transition. Progress is coarse stage
progress: queued `0`, validating `5`, MoGe `10`, compiling `60`, publishing
`95`, and terminal states `100`.

The start route validates the workspace revision, export archive SHA-256,
managed export artifacts, export object ID set, source image artifact, duplicate
active jobs for the workspace/export pair, and global queue limit. Queue
admission is protected by a root-scoped coordination lock shared by job stores
that point at the same workspace root. Queue-wide scans acquire locks in a fixed
order: queue coordination lock first, then workspace locks in sorted workspace
ID order. Stale exports are allowed because exports are immutable; the job
records `export_was_stale_at_start`.

Execution is serialized by an in-process worker with one global reconstruction
worker. API requests return after queueing. Jobs are persisted before background
execution. If worker submission fails after persistence, the queued job is
immediately marked `failed` with a controlled retryable error and the API
returns `503 Service Unavailable`. If a queued worker future is cancelled before
it starts, the job is marked `interrupted`. Queued/running jobs found after
application startup are also marked `interrupted` rather than replayed. Retry is
creating a new job. Active queued/running jobs block workspace deletion with
`409 Conflict`; terminal jobs do not. Workspace deletion fails closed when a
stored reconstruction job record cannot be read or validated, because the active
job state cannot be trusted.

The runner calls MoGe through the existing isolated worker using an internal
trusted-path method against the workspace-managed source image. It always
requests the output set needed by the compiler: `geometry_npz`, `points`,
`depth`, `normal`, `mask`, `intrinsics`, and `previews`. The legacy
`run_inference(image_id=...)` and `/api/moge/infer` path remain unchanged.

After MoGe, the runner invokes the generic compiler without a shell:

```text
<RECONSTRUCTION_PYTHON> -m experiments.moge_scene_graph_spike.src.compile_unified_v3_scene
  --mode clean_reconstruction
  --sam-dir <immutable-export-directory>
  --source-image <managed-source-image>
  --moge-dir <job-staging>/moge
  --output-dir <job-staging>/scene
  --scene-id <stable-scene-id>
  --handoff-dir <job-staging>/handoff
```

The compiler must exit `0`, emit exactly one JSON summary, report
`success=true`, preserve the requested scene ID, and produce object IDs exactly
matching export `mask_id` values. `compilation_report.passed=false` is still a
successful reconstruction job; it means the result requires review.

Successful publication writes `reconstruction-result-manifest.json`, a
sanitized `moge-summary.json`, scene outputs, handoff outputs, and artifact-index
entries atomically. If publication fails, the artifact index and output
directory are rolled back and the job is failed. Reconstruction artifact URLs are
served through `/api/segmentation-artifacts/reconstruction-results/...` with
immutable private caching and hash verification.

## Deferred Features

The following remain intentionally outside this contract:

- Candidate-mask quality reports.
- Final-mask finalization.
- Mask finalization.
- Export import through `/api/scenes/import`.
- Blender integration.
