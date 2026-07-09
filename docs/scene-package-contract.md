# Shared scene-package contract

## Repository architecture

The application is a small FastAPI backend in `backend/` with Pydantic request
models and filesystem-backed SAM operations. The browser client is static HTML,
CSS, and JavaScript in `frontend/`. SAM final exports are written under
`data/exports/`; each export contains binary masks, optional overlays, and a
`metadata.json` manifest.

MoGe and scene-graph work is isolated in
`experiments/moge_scene_graph_spike/`. Its Unified V3.1.1 clean compiler emits
room proxies, camera candidates, object transforms, support relationships,
occlusion evidence, and confidence reports. Blender currently consumes a
generated `blender_one_batch_manifest.json`. There is no installable Blender
addon in this repository; Blender-related files are handoff manifests,
schemas, execution instructions, checkpoints, and validation outputs.

The shared contract lives in `backend/scene_package/` so the backend can own
serialization and validation without coupling it to an API endpoint, frontend
view, Blender transport, or asset search implementation.

## Contract and identity

`Scene` is the JSON document root. Its current `schema_version` is `1.0.0` and
`package_revision` is the optimistic concurrency token for the whole package.
Every `SemanticObject` also has its own monotonic `version`.

The existing final SAM `mask_id` is the stable object UUID. The adapter copies
it byte-for-byte into `object_id`; semantic labels, Blender names, and future
asset identifiers are mutable attributes and never become identity keys.
Mask revision IDs are deterministic UUIDv5 values derived from object ID,
binary-mask SHA-256, and revision number. A mask edit creates a new
`MaskRevision`; it does not create a new semantic object.

Transforms use canonical, right-handed, Z-up world coordinates in meters.
`center` and `dimensions` are XYZ arrays. `quaternion` is ordered WXYZ and must
be normalized. The package explicitly describes canonical-world, image-pixel,
and raw-MoGe coordinate systems. Camera candidates carry world transforms and
normalized intrinsics. `selected_camera` must reference one candidate.

Support targets are checked when the model is constructed:

- `object` targets must identify another semantic object;
- `floor`, `wall`, and `ceiling` targets must identify a matching structural
  room proxy;
- `unknown` support has no target;
- self-support and duplicate object IDs are rejected.

Current mask-revision references, revision parent ownership, duplicate
revision numbers, selected-camera references, and mask-hash agreement are also
validated.

## Artifact boundary

Browser-facing packages never contain filesystem paths. Images, masks,
overlays, manifests, MoGe geometry, renders, and future Blender files are
represented by stable artifact IDs and application URLs under `/artifacts/`
(or an absolute HTTP(S) URL). `mask_filename` is a basename for export/import
compatibility, not a path. Validation rejects absolute filesystem paths and
`file://` URLs, including values nested in metadata.

Artifact URLs are resolved through a server-owned index. Client path fragments
are never joined directly to the filesystem. Both `/artifacts/...` (the URLs
stored in packages) and `/api/artifacts/...` are supported.

## HTTP API

All mutations use optimistic concurrency. A stale package revision, object
version, or mask-revision ID returns `409 Conflict`; clients must reload before
retrying.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/scenes/import` | Import a SAM ZIP, Unified V3 JSON, and source image |
| `GET` | `/api/scenes/{scene_id}` | Return the complete browser-safe package |
| `GET` | `/api/scenes/{scene_id}/objects/{object_id}` | Return one semantic object |
| `GET` | `/api/scenes/{scene_id}/objects/{object_id}/mask` | Return its current PNG mask |
| `GET` | `/api/scenes/{scene_id}/objects/{object_id}/overlay` | Return its current preview |
| `PATCH` | `/api/scenes/{scene_id}/objects/{object_id}/label` | Change semantic/display labels |
| `PATCH` | `/api/scenes/{scene_id}/objects/{object_id}/approval` | Approve, reject, or request revision |
| `POST` | `/api/scenes/{scene_id}/objects/{object_id}/mask-revisions` | Store a delta and immutable resulting mask |
| `GET` | `/api/scenes/{scene_id}/objects/{object_id}/mask-revisions` | Return ordered revision history |
| `PUT` | `/api/scenes/{scene_id}/scene-manifest` | Import newly compiled primitive transforms |
| `GET` | `/api/scenes/{scene_id}/reconstruction-status` | Return stages and stale object IDs |
| `GET` | `/artifacts/{kind}/{artifact_id}` | Serve a registered artifact |

### Import after an MCP masking export

The masking agent finishes the export completely before calling the backend.
Its export directory must contain `metadata.json`, every referenced binary PNG
mask, referenced previews, and any quality report. Zip that directory without
changing its relative paths. Then submit it with the Unified V3 manifest and
the exact source image:

```http
POST /api/scenes/import
Content-Type: multipart/form-data

sam_export=@office-sam.zip;type=application/zip
unified_manifest=@unified_scene_plan.json;type=application/json
source_image=@office.jpg;type=image/jpeg
```

Segmentation workspace export snapshots use the same final-SAM `metadata.json`
shape and ZIP member layout. Their `mask_id` values are the stable segmentation
workspace `object_id` values, so the adapter can consume an extracted workspace
export alongside a matching Unified clean-reconstruction manifest without a
separate adapter path.

The server validates the archive boundary, IDs, one-to-one SAM/Unified object
mapping, actual image types and dimensions, binary mask pixels, and hashes. It
copies accepted inputs into managed storage and returns the new scene package
with status `201`. Import is create-only: an existing `scene_id` returns `409`.
A successful response is the complete `Scene` contract, for example:

```json
{
  "schema_version": "1.0.0",
  "package_revision": 1,
  "scene_id": "office_test_unified_v3_1_1_clean",
  "source_image": {
    "artifact_id": "source-image:24457ea245d9417484c8bc2a235fea3c",
    "url": "/artifacts/source-images/24457ea245d9417484c8bc2a235fea3c"
  },
  "semantic_objects": ["..."],
  "mask_revisions": ["..."]
}
```

The abbreviated array values above are explanatory only; the actual response
contains full object and revision records and validates against the checked-in
JSON Schema. Mutation responses return that same complete package with
incremented `package_revision` and object `version` values.

### Label and approval examples

```json
PATCH /api/scenes/office_test_unified_v3_1_1_clean/objects/d2582b5deecf41cfafa4c10ed0a244f1/label
{
  "semantic_label": "notice_board",
  "display_name": "Notice Board",
  "expected_package_revision": 1,
  "expected_object_version": 1
}
```

```json
PATCH /api/scenes/office_test_unified_v3_1_1_clean/objects/d2582b5deecf41cfafa4c10ed0a244f1/approval
{
  "approval_status": "approved",
  "expected_package_revision": 2,
  "expected_object_version": 2
}
```

### Mask edit example

Submit multipart fields `resulting_mask` (`image/png`), `edit_delta`
(`application/json`), `author`, `operation`, `expected_package_revision`,
`expected_object_version`, and `expected_mask_revision` to the mask-revisions
endpoint. The backend computes the hash and quality measurements from the
pixels, appends revision N+1, preserves revision N and its artifacts, marks the
current object geometry `recompute_required`, and marks primitive/Blender
generation stages `stale`.

The delta is an application-defined JSON object. For brush editing a typical
value is:

```json
{
  "tool": "brush",
  "strokes": [{"mode": "add", "radius": 12, "points": [[101, 82], [104, 84]]}]
}
```

### Compiled transform example

```json
PUT /api/scenes/office_test_unified_v3_1_1_clean/scene-manifest
{
  "expected_package_revision": 3,
  "transforms": [{
    "object_id": "d2582b5deecf41cfafa4c10ed0a244f1",
    "expected_object_version": 3,
    "expected_mask_revision": "68ebad4b23c45eb98ce8ad860d30bd29",
    "transform": {
      "center": [0.1, 2.0, 1.2],
      "dimensions": [0.8, 0.04, 0.5],
      "quaternion": [1.0, 0.0, 0.0, 0.0]
    },
    "blender_object_identifier": "Office.NoticeBoard"
  }]
}
```

Transform imports must name the exact current mask revision. This prevents a
compiler result derived from an older mask from clearing current staleness.

## Storage layout

The default root is `data/scene_packages` and can be changed with
`SCENE_PACKAGE_ROOT`.

```text
data/scene_packages/{scene_id}/
  scene.json
  artifact-index.json
  artifacts/
    source/source.jpg
    inputs/sam-metadata.json
    inputs/unified-scene-plan.json
    inputs/mask-quality-report.json
    revisions/{object_id}/{revision_id}/
      mask.png
      overlay.png              # initial revision when exported
      delta.json               # edited revisions
```

`scene.json`, deltas, and the artifact index use temporary-file + atomic rename
writes. Revision directories and IDs are create-only; previous masks and deltas
are never overwritten.

## Lifecycle

1. A source image receives an image artifact ID.
2. SAM creates a semantic object UUID and initial binary-mask revision.
3. Mask approval or editing appends a revision. Any pixel change marks derived
   geometry `invalidated` or `recompute_required`.
4. MoGe geometry and the Unified compiler populate the canonical primitive
   transform, support, occlusion, and confidence fields.
5. Blender creates or updates an object while preserving `object_id`, then
   writes `blender_object_identifier`.
6. Asset replacement writes `selected_asset_identifier` and changes geometry
   representation without changing `object_id`, mask history, or support
   relationships.

Before a write, callers provide the expected scene `package_revision` and, for
fine-grained changes, expected object versions and current mask-revision IDs.
`validate_optimistic_lock` raises `RevisionConflictError` on stale input. A
successful persistence layer update must increment the affected object
versions and the package revision atomically.

## Serialization and migrations

`dump_scene` emits JSON and `load_scene` migrates before validation. Migrations
are ordered and explicit in `migrations.py`; source mappings are copied and
never modified. Version `0.9.0` documents are supported as the pre-release
field-name migration. Unknown versions fail closed. Future schema changes add
one migration step and retain older steps until stored packages have been
upgraded.

The checked-in JSON Schema is generated from the Pydantic models using Draft
2020-12. Schema and application-model validation are both exercised by tests.

## Adapter and fixture

`adapt_scene_package` accepts final SAM `metadata.json`, its sibling masks, and
a Unified V3.1.1 clean `unified_scene_plan.json`. It verifies a one-to-one ID
match, hashes mask bytes, maps geometry/camera/room evidence, generates stable
revision IDs, and returns a fully validated `Scene`. Input artifacts are read
only.

The office fixture was generated from the existing final office SAM export and
`unified_v3_1_1_clean/unified_scene_plan.json`. It contains 20 objects, 20 mask
revisions, three room proxies, and two camera candidates.

## Deferred integration decisions

- Blender object naming/ID property conventions and addon transport are not
  defined because no addon exists here.
- Mask approval roles and authorization policy remain application concerns.
- Asset-catalog identifier namespace, licensing metadata, and replacement
  transform policy remain open.
- Whether Blender writes back its exact evaluated transform or a separate
  render/export transform needs an integration decision.
