import test from "node:test";
import assert from "node:assert/strict";
import {SegmentationWorkspaceClient} from "../../frontend/segment-api.js";
import {
  ReconstructionWorkspaceState,
  createReconstructionRequestSnapshot,
  hidePathLikeValue,
  isActiveReconstructionJob,
  isTerminalReconstructionJob,
  normalizeCompilationReport,
  normalizeMogeSummary,
  normalizeReconstructionHealth,
  normalizeReconstructionJob,
  reconstructionOutcomeLabel,
  reconstructionRequestPayload,
  reconstructionStartReadiness,
  sortReconstructionJobsNewestFirst
} from "../../frontend/segment-reconstructions.js";

const sha="a".repeat(64);
const workspace={workspace_id:"workspace-a",workspace_revision:18};
const exportRecord={export_id:"export-a",archive_sha256:sha,is_stale:false};
const readyHealth={configured:true,moge_configured:true,compiler_configured:true,worker_running:false,device:"cuda",model:"moge-2-vitl-normal",queue_running:0,queue_queued:0,error:null};

function job(overrides={}){
  return {
    job_id:"job-a",
    job_version:1,
    workspace_id:"workspace-a",
    export_id:"export-a",
    export_archive_sha256:sha,
    source_image_id:"image-a",
    status:"queued",
    stage:"queued",
    progress_percent:0,
    created_at:"2026-07-10T08:00:00Z",
    started_at:null,
    updated_at:"2026-07-10T08:00:00Z",
    finished_at:null,
    request:{expected_workspace_revision:18,expected_export_archive_sha256:sha,resolution_level:9,num_tokens:null},
    export_was_stale_at_start:false,
    semantic_object_count:1,
    object_ids:["object-a"],
    scene_id:"scene-a",
    result:null,
    error:null,
    ...overrides
  };
}

function succeeded(overrides={}){
  return job({
    status:"succeeded",
    stage:"complete",
    progress_percent:100,
    finished_at:"2026-07-10T08:03:00Z",
    result:{
      scene_id:"scene-a",
      semantic_object_count:1,
      object_ids:["object-a"],
      compilation_passed:false,
      artifacts:{
        result_manifest_url:"/api/segmentation-artifacts/reconstruction-results/w/j/result-manifest",
        unified_scene_plan_url:"/api/segmentation-artifacts/reconstruction-results/w/j/scene-plan",
        compilation_report_url:"/api/segmentation-artifacts/reconstruction-results/w/j/compilation-report",
        compilation_markdown_url:"/api/segmentation-artifacts/reconstruction-results/w/j/compilation-markdown",
        room_plan_url:"/api/segmentation-artifacts/reconstruction-results/w/j/room-plan",
        camera_candidates_url:"/api/segmentation-artifacts/reconstruction-results/w/j/camera-candidates",
        object_pose_report_url:"/api/segmentation-artifacts/reconstruction-results/w/j/object-pose-report",
        placement_report_url:"/api/segmentation-artifacts/reconstruction-results/w/j/placement-report",
        collision_report_url:"/api/segmentation-artifacts/reconstruction-results/w/j/collision-report",
        confidence_report_url:"/api/segmentation-artifacts/reconstruction-results/w/j/confidence-report",
        support_graph_url:"/api/segmentation-artifacts/reconstruction-results/w/j/support-graph",
        blender_manifest_url:"/api/segmentation-artifacts/reconstruction-results/w/j/blender-manifest",
        moge_geometry_url:"/api/segmentation-artifacts/reconstruction-results/w/j/moge-geometry",
        moge_summary_url:"/api/segmentation-artifacts/reconstruction-results/w/j/moge-summary",
        overview_url:null
      }
    },
    ...overrides
  });
}

test("health normalization and readiness block unavailable services",()=>{
  assert.equal(normalizeReconstructionHealth({...readyHealth,device:"C:\\secret"}).device,null);
  assert.equal(reconstructionStartReadiness({workspace,exportRecord,health:normalizeReconstructionHealth(readyHealth)}).ready,true);
  assert.equal(reconstructionStartReadiness({workspace,exportRecord,health:null}).reasons[0].code,"health_missing");
  assert.equal(reconstructionStartReadiness({workspace,exportRecord,health:normalizeReconstructionHealth({...readyHealth,configured:false,moge_configured:false})}).ready,false);
  assert.equal(reconstructionStartReadiness({workspace,exportRecord,health:normalizeReconstructionHealth({...readyHealth,configured:false,compiler_configured:false})}).ready,false);
  assert.equal(reconstructionStartReadiness({workspace,exportRecord:{...exportRecord,is_stale:true},health:normalizeReconstructionHealth(readyHealth)}).ready,true);
  assert.equal(reconstructionStartReadiness({workspace,exportRecord,health:normalizeReconstructionHealth(readyHealth),exportCreationActive:true}).reasons[0].code,"export_active");
  assert.equal(reconstructionStartReadiness({workspace,exportRecord,health:normalizeReconstructionHealth(readyHealth),startActive:true}).reasons[0].code,"start_active");
  assert.equal(reconstructionStartReadiness({workspace,exportRecord,health:normalizeReconstructionHealth(readyHealth),resolutionLevel:10}).reasons[0].code,"invalid_resolution");
  assert.equal(reconstructionStartReadiness({workspace,exportRecord,health:normalizeReconstructionHealth(readyHealth),numTokensInput:"1.5"}).reasons[0].code,"invalid_tokens");
});

test("request snapshot captures immutable values and payload sends null tokens",()=>{
  const mutableWorkspace={...workspace};
  const mutableExport={...exportRecord};
  const snapshot=createReconstructionRequestSnapshot({operationId:7,workspace:mutableWorkspace,exportRecord:mutableExport,resolutionLevel:9,numTokens:"",generation:3});
  mutableWorkspace.workspace_id="other";
  mutableExport.archive_sha256="b".repeat(64);
  assert.deepEqual(snapshot,{operationId:7,workspaceId:"workspace-a",expectedWorkspaceRevision:18,exportId:"export-a",expectedExportArchiveSha256:sha,resolutionLevel:9,numTokens:null,generation:3});
  assert.deepEqual(reconstructionRequestPayload(snapshot),{expected_workspace_revision:18,expected_export_archive_sha256:sha,resolution_level:9,num_tokens:null});
});

test("job normalization preserves statuses, false compilation, artifacts, and errors",()=>{
  assert.equal(normalizeReconstructionJob(job({status:"new-status",stage:"new-stage"})).status,"unknown");
  assert.equal(isActiveReconstructionJob(normalizeReconstructionJob(job({status:"running",stage:"moge",progress_percent:10}))),true);
  assert.equal(isTerminalReconstructionJob(normalizeReconstructionJob(job({status:"failed",stage:"failed",progress_percent:100,finished_at:"now",error:{code:"x",message:"failed",stage:"failed",retryable:true}}))),true);
  const normalized=normalizeReconstructionJob(succeeded());
  assert.equal(normalized.result.compilation_passed,false);
  assert.equal(normalized.result.artifacts.overview_url,null);
  assert.equal(reconstructionOutcomeLabel(normalized),"Succeeded - review required");
  const failed=normalizeReconstructionJob(job({status:"failed",stage:"failed",progress_percent:100,finished_at:"now",error:{code:"queue_limit",message:"queue limit reached",stage:"failed",retryable:false}}));
  assert.equal(failed.error.code,"queue_limit");
  assert.equal(failed.error.retryable,false);
});

test("job sorting and version-aware authoritative merging protect newer terminal records",()=>{
  const state=new ReconstructionWorkspaceState({setTimeoutFn:()=>1,clearTimeoutFn:()=>{}});
  state.reset("workspace-a");
  const context=state.beginJobs("workspace-a");
  state.setJobs(context,[job({job_id:"older",created_at:"2026-07-09T00:00:00Z"}),job({job_id:"newer",created_at:"2026-07-10T00:00:00Z"})]);
  assert.deepEqual(state.jobs.map(item=>item.job_id),["newer","older"]);
  state.upsertJob(succeeded({job_id:"newer",job_version:5}));
  const staleContext=state.beginJobs("workspace-a");
  state.setJobs(staleContext,[job({job_id:"newer",job_version:4,status:"running",stage:"moge",progress_percent:10,started_at:"now"}),job({job_id:"retry",created_at:"2026-07-11T00:00:00Z"})]);
  assert.equal(state.jobs.find(item=>item.job_id==="newer").job_version,5);
  assert.equal(state.jobs.find(item=>item.job_id==="newer").status,"succeeded");
  assert.deepEqual(state.jobs.map(item=>item.job_id),["retry","newer"]);
  assert.deepEqual(sortReconstructionJobsNewestFirst([job({job_id:"a",created_at:"2026-07-08"}),job({job_id:"b",created_at:"2026-07-09"})]).map(item=>item.job_id),["b","a"]);
});

test("state guards late health, job, start, poll, and diagnostic responses",()=>{
  const state=new ReconstructionWorkspaceState({setTimeoutFn:()=>42,clearTimeoutFn:()=>{}});
  const healthA=state.beginHealth("workspace-a");
  state.reset("workspace-b");
  assert.equal(state.setHealth(healthA,readyHealth),false);
  const jobsB=state.beginJobs("workspace-b");
  state.setJobs(jobsB,[job({workspace_id:"workspace-b"}),job({job_id:"job-b",workspace_id:"workspace-b"})]);
  const jobsA={workspaceId:"workspace-a",generation:jobsB.generation,pollGeneration:state.pollGeneration};
  assert.equal(state.setJobs(jobsA,[job()]),false);
  const snapshot=state.startSnapshot({workspace_id:"workspace-b",workspace_revision:2},{export_id:"export-b",archive_sha256:sha},{resolutionLevel:9,numTokens:null});
  assert.equal(state.beginStart(snapshot),true);
  assert.equal(state.beginStart(snapshot),false);
  state.finishStart(snapshot);
  state.schedulePoll(()=>{},2000);
  assert.equal(state.pollTimer,42);
  const selected=state.jobs[0];
  state.selectJob(selected.job_id);
  const diag=state.beginDiagnostic(selected,"/api/segmentation-artifacts/reconstruction-results/w/j/compilation-report","compilation");
  state.selectJob(state.jobs.find(item=>item.job_id!==selected.job_id).job_id);
  assert.equal(state.setDiagnostic(diag,{passed:true}),false);
});

test("poll failure backoff is bounded and manual refresh resets it",()=>{
  const state=new ReconstructionWorkspaceState();
  state.reset("workspace-a");
  const context=state.beginJobs("workspace-a");
  state.failJobs(context,new Error("network"),{poll:true});
  assert.equal(state.pollDelay(),4000);
  for(let i=0;i<8;i++){
    const c=state.beginJobs("workspace-a");
    state.failJobs(c,new Error("network"),{poll:true});
  }
  assert.equal(state.pollDelay(),15000);
  state.beginJobs("workspace-a",{manual:true});
  assert.equal(state.pollFailureCount,0);
});

test("retry readiness uses a new request with original export and current workspace revision",()=>{
  const failed=normalizeReconstructionJob(job({status:"failed",stage:"failed",progress_percent:100,finished_at:"now",request:{expected_workspace_revision:8,expected_export_archive_sha256:sha,resolution_level:6,num_tokens:128},error:{code:"executor",message:"failed",stage:"failed",retryable:true}}));
  const retryExport={export_id:failed.export_id,archive_sha256:"b".repeat(64)};
  const snapshot=createReconstructionRequestSnapshot({operationId:2,workspace:{workspace_id:failed.workspace_id,workspace_revision:20},exportRecord:retryExport,resolutionLevel:failed.request.resolution_level,numTokens:failed.request.num_tokens,generation:1});
  assert.equal(snapshot.expectedWorkspaceRevision,20);
  assert.equal(snapshot.exportId,failed.export_id);
  assert.equal(snapshot.expectedExportArchiveSha256,"b".repeat(64));
  assert.equal(snapshot.numTokens,128);
});

test("diagnostic normalization is dynamic and hides path-like values",()=>{
  const report=normalizeCompilationReport({passed:false,object_count:2,quality_gates:{object_count_matches_export:true,no_semantic_collisions:false},confidence_counts:{high:1},placement_counts:{floor:2},clean_scene_plan_sha256:sha});
  assert.deepEqual(report.quality_gates.map(item=>[item.key,item.passed]),[["object_count_matches_export",true],["no_semantic_collisions",false]]);
  const summary=normalizeMogeSummary({model:"moge",source_image:"/tmp/source.png",gpu:{name:"RTX",path:"C:\\secret"},shape:[1,2,3]});
  assert.equal(summary.visible.source_image,"Value hidden");
  assert.equal(summary.visible.gpu.path,"Value hidden");
  assert.equal(hidePathLikeValue("file:///tmp/a"),"Value hidden");
});

test("reconstruction API methods encode ids and keep artifact fetching browser-safe",async()=>{
  const calls=[];
  global.fetch=async(url,options={})=>{calls.push({url,options});return{ok:true,json:async()=>({ok:true})}};
  const api=new SegmentationWorkspaceClient();
  await api.getReconstructionHealth();
  await api.startReconstruction({workspaceId:"workspace/a",exportId:"export/a",expectedWorkspaceRevision:18,expectedExportArchiveSha256:sha,resolutionLevel:9,numTokens:undefined});
  await api.listReconstructions("workspace/a");
  await api.getReconstruction("workspace/a","job/a");
  await api.getJsonArtifact("/api/segmentation-artifacts/reconstruction-results/w/j/compilation-report");
  assert.equal(calls[0].url,"/api/segmentation-reconstruction/health");
  assert.equal(calls[1].url,"/api/segmentation-workspaces/workspace%2Fa/exports/export%2Fa/reconstructions");
  assert.deepEqual(JSON.parse(calls[1].options.body),{expected_workspace_revision:18,expected_export_archive_sha256:sha,resolution_level:9,num_tokens:null});
  assert.equal(calls[2].url,"/api/segmentation-workspaces/workspace%2Fa/reconstructions");
  assert.equal(calls[3].url,"/api/segmentation-workspaces/workspace%2Fa/reconstructions/job%2Fa");
  assert.equal(calls[4].url,"/api/segmentation-artifacts/reconstruction-results/w/j/compilation-report");
  assert.throws(()=>api.getJsonArtifact("C:\\tmp\\geometry.npz"),/Artifact URL must be an application URL/);
});
