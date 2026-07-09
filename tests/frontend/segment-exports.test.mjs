import test from "node:test";
import assert from "node:assert/strict";
import {SegmentationWorkspaceClient} from "../../frontend/segment-api.js";
import {
  ExportWorkspaceState,
  createExportRequestSnapshot,
  exportReadiness,
  exportRecordMatchesWorkspace,
  exportRequestPayload,
  normalizeQualityReport,
  sortExportsNewestFirst,
  sourceKindLabel,
  staleLabel
} from "../../frontend/segment-exports.js";

function candidateWorkspace(overrides={}){
  return {
    workspace_id:"workspace-a",
    workspace_revision:7,
    objects:[{
      object_id:"object-a",
      object_version:3,
      semantic_label:"office_chair",
      display_name:"Office Chair",
      manual_mask:{manual_revision:0},
      sam_draft:{
        prompt_revision:2,
        selected_candidate_index:1,
        candidates:[{candidate_index:1,mask_url:"/api/segmentation-artifacts/sam-candidates/workspace-a/object-a/2/1"}]
      }
    }],
    ...overrides
  };
}

function manualWorkspace(){
  return candidateWorkspace({
    objects:[{
      object_id:"object-a",
      object_version:4,
      semantic_label:"office_chair",
      display_name:"Office Chair",
      manual_mask:{
        manual_revision:2,
        base_prompt_revision:5,
        base_candidate_index:0,
        composite_mask_url:"/api/segmentation-artifacts/manual-masks/workspace-a/object-a/2/composite"
      },
      sam_draft:{prompt_revision:5,selected_candidate_index:0,candidates:[{candidate_index:0,mask_url:"/api/segmentation-artifacts/sam-candidates/workspace-a/object-a/5/0"}]}
    }]
  });
}

function exportRecord(overrides={}){
  return {
    export_id:"export-a",
    created_at:"2026-07-09T08:00:00Z",
    created_from_workspace_revision:7,
    published_workspace_revision:8,
    masks:[{
      object_id:"object-a",
      object_version:3,
      semantic_label:"office_chair",
      display_name:"Office Chair",
      source_kind:"sam_candidate",
      source_prompt_revision:2,
      source_candidate_index:1,
      source_manual_revision:0,
      filename:"office_chair.png",
      mask_url:"/api/segmentation-artifacts/workspace-exports/workspace-a/export-a/masks/object-a",
      preview_url:"/api/segmentation-artifacts/workspace-exports/workspace-a/export-a/previews/object-a",
      mask_sha256:"a".repeat(64),
      area_pixels:12,
      bbox_xyxy:[1,2,3,4]
    }],
    metadata_url:"/api/segmentation-artifacts/workspace-exports/workspace-a/export-a/metadata",
    quality_report_url:"/api/segmentation-artifacts/workspace-exports/workspace-a/export-a/quality-json",
    quality_markdown_url:"/api/segmentation-artifacts/workspace-exports/workspace-a/export-a/quality-markdown",
    combined_preview_url:"/api/segmentation-artifacts/workspace-exports/workspace-a/export-a/combined-preview",
    archive_url:"/api/segmentation-artifacts/workspace-exports/workspace-a/export-a/archive",
    archive_sha256:"b".repeat(64),
    warning_count:0,
    mask_count:1,
    total_mask_area:12,
    is_stale:false,
    ...overrides
  };
}

test("export request snapshot captures immutable revision and object versions",()=>{
  const workspace=candidateWorkspace();
  const snapshot=createExportRequestSnapshot(workspace,9);
  workspace.workspace_revision=99;
  workspace.objects[0].object_version=42;
  assert.deepEqual(snapshot,{
    operationId:9,
    workspaceId:"workspace-a",
    expectedWorkspaceRevision:7,
    expectedObjects:[{object_id:"object-a",expected_object_version:3}]
  });
  assert.deepEqual(exportRequestPayload(snapshot),{
    expected_workspace_revision:7,
    expected_objects:[{object_id:"object-a",expected_object_version:3}],
    include_previews:true
  });
});

test("export readiness requires persisted effective masks and no active work",()=>{
  assert.deepEqual(exportReadiness(null).reasons.map(item=>item.code),["no_workspace"]);
  assert.deepEqual(exportReadiness(candidateWorkspace({objects:[]})).reasons.map(item=>item.code),["no_objects"]);
  assert.equal(exportReadiness(candidateWorkspace()).ready,true);
  assert.equal(exportReadiness(manualWorkspace()).ready,true);
  assert.deepEqual(exportReadiness(candidateWorkspace(),{brushDirty:true,predictionActive:true,exportActive:true}).reasons.map(item=>item.code),["dirty_brush","prediction_active","export_active"]);
  const missing=candidateWorkspace({objects:[{...candidateWorkspace().objects[0],sam_draft:{prompt_revision:0,selected_candidate_index:null,candidates:[]}}]});
  assert.equal(exportReadiness(missing).ready,false);
  assert.equal(exportReadiness(missing).reasons[0].code,"missing_effective_mask");
});

test("export history sorts newest first and reports current/stale labels",()=>{
  const records=sortExportsNewestFirst([
    exportRecord({export_id:"older",created_at:"2026-07-08T08:00:00Z",published_workspace_revision:3,is_stale:true}),
    exportRecord({export_id:"newer",created_at:"2026-07-09T08:00:00Z",published_workspace_revision:4,is_stale:false})
  ]);
  assert.deepEqual(records.map(item=>item.export_id),["newer","older"]);
  assert.equal(staleLabel(records[0]),"Current");
  assert.equal(staleLabel(records[1]),"Stale");
  assert.equal(staleLabel({}),"Unknown");
});

test("export records match the active workspace by stable object identity and effective mask anchors",()=>{
  assert.equal(exportRecordMatchesWorkspace(exportRecord(),candidateWorkspace()),true);
  assert.equal(exportRecordMatchesWorkspace(exportRecord({masks:[{...exportRecord().masks[0],semantic_label:"renamed"}]}),candidateWorkspace()),false);
  assert.equal(exportRecordMatchesWorkspace(exportRecord(),candidateWorkspace({objects:[{...candidateWorkspace().objects[0],object_id:"other"}]})),false);
  const manual=manualWorkspace();
  const record=exportRecord({masks:[{...exportRecord().masks[0],object_version:4,source_kind:"manual_composite",source_prompt_revision:5,source_candidate_index:0,source_manual_revision:2}]});
  assert.equal(exportRecordMatchesWorkspace(record,manual),true);
});

test("export workspace state guards late list, create, selection, and quality results",()=>{
  const state=new ExportWorkspaceState();
  const listGeneration=state.beginList("workspace-a");
  assert.equal(state.isCurrent("workspace-a",listGeneration),true);
  state.reset("workspace-b");
  assert.equal(state.isCurrent("workspace-a",listGeneration),false);
  state.reset("workspace-a");
  const snapshot=state.beginCreate(candidateWorkspace());
  assert.equal(state.beginCreate(candidateWorkspace()),null);
  assert.equal(state.isCurrentOperation(snapshot,"workspace-a"),true);
  state.finishCreate(snapshot);
  const normalized=state.upsertRecord(exportRecord());
  assert.equal(state.selectedRecord().export_id,"export-a");
  const qualityContext=state.beginQuality(normalized);
  state.setQuality(normalized,{summary:{warnings:[{message:"small"}]},masks:[{label:"Office Chair"}]});
  assert.equal(state.cachedQuality(normalized).summary.warnings.length,1);
  state.select("export-a");
  assert.equal(state.isCurrentQuality(qualityContext),true);
  state.reset("workspace-c");
  assert.equal(state.isCurrentQuality(qualityContext),false);
});

test("quality report normalization and source labels tolerate missing optional arrays",()=>{
  const report=normalizeQualityReport({summary:{}});
  assert.deepEqual(report.masks,[]);
  assert.deepEqual(report.pairwise_overlaps,[]);
  assert.deepEqual(report.bbox_comparisons,[]);
  assert.deepEqual(report.summary.warnings,[]);
  assert.equal(sourceKindLabel("manual_composite"),"Saved manual composite");
  assert.equal(sourceKindLabel("sam_candidate"),"SAM candidate");
});

test("segmentation export API methods use backend export contract",async()=>{
  const calls=[];
  global.fetch=async(url,options={})=>{calls.push({url,options});return{ok:true,json:async()=>({ok:true})}};
  const api=new SegmentationWorkspaceClient();
  await api.createExport({workspaceId:"workspace/a",expectedWorkspaceRevision:8,expectedObjects:[{object_id:"object-a",expected_object_version:3}]});
  await api.listExports("workspace/a");
  await api.getExport("workspace/a","export/a");
  await api.getJsonArtifact("/api/segmentation-artifacts/workspace-exports/workspace-a/export-a/quality-json");
  assert.equal(calls[0].url,"/api/segmentation-workspaces/workspace%2Fa/exports");
  assert.deepEqual(JSON.parse(calls[0].options.body),{expected_workspace_revision:8,expected_objects:[{object_id:"object-a",expected_object_version:3}],include_previews:true});
  assert.equal(calls[1].url,"/api/segmentation-workspaces/workspace%2Fa/exports");
  assert.equal(calls[2].url,"/api/segmentation-workspaces/workspace%2Fa/exports/export%2Fa");
  assert.equal(calls[3].url,"/api/segmentation-artifacts/workspace-exports/workspace-a/export-a/quality-json");
  assert.throws(()=>api.getJsonArtifact("file:///tmp/report.json"),/Artifact URL must be an application URL/);
});
