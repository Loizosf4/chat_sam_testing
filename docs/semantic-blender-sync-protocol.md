# Semantic website–Blender synchronization protocol

Status: version 1.0.0. Asset replacement is intentionally out of scope.

## Architecture

The existing transports are retained:

1. The website loads and edits scene packages through the FastAPI HTTP API.
2. The Blender addon retains the reference addon's local HTTP server at `127.0.0.1:8765`.
3. The website sends versioned JSON messages to `POST /command` and polls `GET /events?since=<sequence>` for Blender-originated events.

The addon owns only objects carrying `sam_semantic_id` and `sam_scene_id`. It creates them in `SAM Scene <scene_id>`. Objects without those properties are never deleted, moved, renamed, or relinked. The stable scene-package `object_id` is stored as `sam_semantic_id`; that same ID is used by the website mask and is reserved for a future replacement model.

Semantic objects and structural room proxies are unit cubes whose location, dimensions, and `wxyz` quaternion come from the scene package. A semantic revision token combines object version, mask revision, and transform digest. A room-proxy revision token is its transform digest. Matching tokens are left untouched; changed tokens are updated in place; absent IDs are created once. Duplicate IDs cause synchronization to fail.

Blender persists `sam_scene_id`, `sam_scene_revision`, `sam_sync_status`, and JSON `sam_sync_details` on the Blender scene. Each managed object persists its kind and revision; semantic objects also persist object version, mask revision, and semantic label.

## HTTP endpoints

- `GET /health` returns addon availability and protocol version.
- `POST /command` accepts one message envelope and returns `sync_ack` or `error`.
- `GET /events?since=N` returns `{schema_version, latest_sequence, events}`. The bounded journal holds the latest 512 events.

The server binds only to `127.0.0.1`, accepts JSON, limits commands to 8 MiB, queues Blender mutations onto Blender's main thread, and supplies CORS headers for the local website.

## Envelope

```json
{
  "schema_version": "1.0.0",
  "message_id": "uuid",
  "correlation_id": "request uuid on responses",
  "type": "handshake",
  "scene_id": "office_test_unified_v3_1_1_clean",
  "object_id": "d2582b5deecf41cfafa4c10ed0a244f1",
  "revision": 7,
  "payload": {}
}
```

`scene_id`, `object_id`, and `revision` may be `null` only when the message type does not need them. Object-scoped messages require all three. Unknown message types and schema versions are rejected.

## Message types

| Type | Direction | Required payload / behavior |
|---|---|---|
| `handshake` | website → Blender | Negotiates schema and returns addon version, capabilities, and saved synchronization status. |
| `scene_sync` | website → Blender | `scene_package` plus `stale_object_ids`; creates or updates room proxies and semantic cubes by stable ID. |
| `select_highlight` | website → Blender | Envelope `object_id`; payload `mask_revision`; selects and highlights the mapped Blender object. |
| `selection_changed` | Blender → website | Object ID, object/mask revisions, and current transform. The website selects the matching mask. |
| `transform_update` | Blender → website | Object ID, object/mask revisions, and `{center, dimensions, quaternion}`. The website writes it through the revision-checked scene-manifest API. |
| `sync_ack` | either direction | Correlates to the triggering message and records acceptance or created/updated/unchanged IDs. |
| `error` | Blender → website | Structured error payload described below. |

The reference addon's `select_object`, `highlight_object`, `get_selected_object`, and `get_scene_state` command names remain accepted as compatibility adapters. Per-object creation commands are superseded by atomic `scene_sync` so duplicate and revision checks apply consistently.

## Error response

```json
{
  "type": "error",
  "correlation_id": "request uuid",
  "payload": {
    "code": "stale_mask_revision",
    "message": "website mask revision differs from Blender",
    "details": {"blender_mask_revision": "..."},
    "retryable": false
  }
}
```

Defined failure codes include `schema_version_mismatch`, `invalid_scene_package`, `duplicate_semantic_id`, `scene_mismatch`, `stale_revision`, `stale_mask_revision`, `mapping_not_found`, `timeout`, and `internal_error`.

## Revision and reconnect rules

- Blender rejects a `scene_sync` revision older than its saved revision for that scene.
- Selecting with a mask revision different from the mapped Blender object's revision fails with `stale_mask_revision`.
- The website derives stale geometry from current mask revisions whose `geometry_invalidation_status` is `invalidated` or `recompute_required`; it shows a stale connection state and sends those IDs during synchronization.
- HTTP 409 from a Blender transform write means the scene/object/mask revision is stale. The website reports it and does not overwrite current data.
- Loading a scene disconnects the prior event poll, performs a new handshake and full ID/revision reconciliation, then restarts polling from the last event sequence. The retry action performs the same handshake and reconciliation.

## Installation and demonstration

Install the generated `sam-semantic-blender-addon.zip` through Blender Preferences → Add-ons → Install from Disk, then enable **SAM Semantic Scene Bridge**. Start the FastAPI application and load `office_test_unified_v3_1_1_clean` in the website. Clicking a mask selects the matching cube. Selecting or transforming a managed cube in Blender updates the website. The top-bar indicator and error banner show connection state and failures.

The accepted-office validation artifact is `docs/demonstration/accepted-office-v3.1.1-sync.json`.
