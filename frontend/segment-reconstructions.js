const KNOWN_STATUSES=new Set(["queued","running","succeeded","failed","interrupted"]);
const KNOWN_STAGES=new Set(["queued","validating","moge","compiling","publishing","complete","failed","interrupted"]);
const TERMINAL_STATUSES=new Set(["succeeded","failed","interrupted"]);
const ACTIVE_STATUSES=new Set(["queued","running"]);
const ARTIFACT_KEYS=[
  "result_manifest_url","unified_scene_plan_url","compilation_report_url","compilation_markdown_url",
  "room_plan_url","camera_candidates_url","object_pose_report_url","placement_report_url",
  "collision_report_url","confidence_report_url","support_graph_url","overview_url",
  "projected_primitives_url","room_camera_url","confidence_overview_url","ambiguity_overview_url",
  "blender_manifest_url","moge_geometry_url","moge_summary_url","depth_preview_url",
  "normal_preview_url","valid_mask_preview_url"
];
const PATH_LIKE=/^([A-Za-z]:[\\/]|\/|file:\/\/)/;

function stringOrNull(value){return value===null||value===undefined?null:String(value)}
function numberOrNull(value){const number=Number(value);return Number.isFinite(number)?number:null}
function integerOrNull(value){const number=Number(value);return Number.isInteger(number)?number:null}
function boolOrNull(value){return typeof value==="boolean"?value:null}
function timestampOrNull(value){return value===null||value===undefined?null:String(value)}
function safeUrl(value){return value===null||value===undefined||value===""?null:String(value)}
function normalizeStatus(value){const status=String(value||"unknown");return KNOWN_STATUSES.has(status)?status:"unknown"}
function normalizeStage(value){const stage=String(value||"unknown");return KNOWN_STAGES.has(stage)?stage:"unknown"}

export function hidePathLikeValue(value){
  return typeof value==="string"&&PATH_LIKE.test(value)?"Value hidden":value;
}

export function normalizeReconstructionHealth(health={}){
  return {
    configured:typeof health.configured==="boolean"?health.configured:false,
    moge_configured:typeof health.moge_configured==="boolean"?health.moge_configured:false,
    compiler_configured:typeof health.compiler_configured==="boolean"?health.compiler_configured:false,
    worker_running:typeof health.worker_running==="boolean"?health.worker_running:false,
    device:typeof health.device==="string"&&!PATH_LIKE.test(health.device)?health.device:null,
    model:typeof health.model==="string"&&!PATH_LIKE.test(health.model)?health.model:null,
    queue_running:Number.isFinite(Number(health.queue_running))?Math.max(0,Number(health.queue_running)):0,
    queue_queued:Number.isFinite(Number(health.queue_queued))?Math.max(0,Number(health.queue_queued)):0,
    error:health.error?String(health.error):null
  };
}

export function normalizeReconstructionArtifacts(artifacts={}){
  const normalized={};
  for(const key of ARTIFACT_KEYS)normalized[key]=safeUrl(artifacts?.[key]);
  return normalized;
}

export function normalizeReconstructionResult(result=null){
  if(!result)return null;
  return {
    scene_id:stringOrNull(result.scene_id),
    semantic_object_count:Number(result.semantic_object_count||0),
    object_ids:Array.isArray(result.object_ids)?result.object_ids.map(String):[],
    compilation_passed:boolOrNull(result.compilation_passed),
    artifacts:normalizeReconstructionArtifacts(result.artifacts||{})
  };
}

export function normalizeReconstructionError(error=null){
  if(!error)return null;
  return {
    code:String(error.code||"unknown_error"),
    message:String(error.message||"Reconstruction failed."),
    stage:normalizeStage(error.stage),
    retryable:typeof error.retryable==="boolean"?error.retryable:false
  };
}

export function normalizeReconstructionJob(job={}){
  const result=normalizeReconstructionResult(job.result);
  const status=normalizeStatus(job.status);
  const stage=normalizeStage(job.stage);
  return {
    ...job,
    job_id:String(job.job_id||""),
    job_version:Number(job.job_version||0),
    workspace_id:String(job.workspace_id||""),
    export_id:String(job.export_id||""),
    export_archive_sha256:String(job.export_archive_sha256||""),
    source_image_id:String(job.source_image_id||""),
    status,
    stage,
    progress_percent:Number.isFinite(Number(job.progress_percent))?Math.min(100,Math.max(0,Number(job.progress_percent))):0,
    created_at:timestampOrNull(job.created_at),
    started_at:timestampOrNull(job.started_at),
    updated_at:timestampOrNull(job.updated_at),
    finished_at:timestampOrNull(job.finished_at),
    request:{
      expected_workspace_revision:Number(job.request?.expected_workspace_revision||0),
      expected_export_archive_sha256:String(job.request?.expected_export_archive_sha256||""),
      resolution_level:Number(job.request?.resolution_level||0),
      num_tokens:job.request?.num_tokens===null||job.request?.num_tokens===undefined?null:Number(job.request.num_tokens)
    },
    export_was_stale_at_start:typeof job.export_was_stale_at_start==="boolean"?job.export_was_stale_at_start:false,
    semantic_object_count:Number(job.semantic_object_count||result?.semantic_object_count||0),
    object_ids:Array.isArray(job.object_ids)?job.object_ids.map(String):result?.object_ids||[],
    scene_id:String(job.scene_id||result?.scene_id||""),
    result,
    error:normalizeReconstructionError(job.error)
  };
}

export function sortReconstructionJobsNewestFirst(jobs=[]){
  return [...jobs].map(normalizeReconstructionJob).sort((a,b)=>{
    const byDate=Date.parse(b.created_at||"")-Date.parse(a.created_at||"");
    if(Number.isFinite(byDate)&&byDate!==0)return byDate;
    return String(b.job_id).localeCompare(String(a.job_id));
  });
}

export function isActiveReconstructionJob(job){return ACTIVE_STATUSES.has(job?.status)}
export function isTerminalReconstructionJob(job){return TERMINAL_STATUSES.has(job?.status)}

export function reconstructionStatusLabel(status){
  return ({queued:"Queued",running:"Running",succeeded:"Succeeded",failed:"Failed",interrupted:"Interrupted"})[status]||"Unknown";
}

export function reconstructionStageLabel(stage){
  return ({
    queued:"Waiting in queue",
    validating:"Validating immutable inputs",
    moge:"Running MoGe reconstruction",
    compiling:"Compiling scene geometry",
    publishing:"Publishing immutable results",
    complete:"Complete",
    failed:"Failed",
    interrupted:"Interrupted"
  })[stage]||"Unknown";
}

export function reconstructionOutcomeLabel(job){
  if(job?.status==="succeeded"&&job.result?.compilation_passed===true)return "Succeeded - passed";
  if(job?.status==="succeeded"&&job.result?.compilation_passed===false)return "Succeeded - review required";
  return reconstructionStatusLabel(job?.status);
}

export function createReconstructionRequestSnapshot({
  operationId,
  workspace,
  exportRecord,
  resolutionLevel,
  numTokens,
  stateGeneration
}){
  return {
    operationId:Number(operationId),
    workspaceId:String(workspace.workspace_id),
    expectedWorkspaceRevision:Number(workspace.workspace_revision),
    exportId:String(exportRecord.export_id),
    expectedExportArchiveSha256:String(exportRecord.archive_sha256),
    resolutionLevel:Number(resolutionLevel),
    numTokens:numTokens===null||numTokens===undefined||numTokens===""?null:Number(numTokens),
    stateGeneration:Number(stateGeneration)
  };
}

export function reconstructionRequestPayload(snapshot){
  return {
    expected_workspace_revision:snapshot.expectedWorkspaceRevision,
    expected_export_archive_sha256:snapshot.expectedExportArchiveSha256,
    resolution_level:snapshot.resolutionLevel,
    num_tokens:snapshot.numTokens===null||snapshot.numTokens===undefined?null:snapshot.numTokens
  };
}

export function reconstructionStartReadiness({
  workspace,
  exportRecord,
  health,
  startActive=false,
  exportCreationActive=false,
  resolutionLevel=9,
  numTokensInput=""
}={}){
  const reasons=[];
  const resolution=Number(resolutionLevel);
  const tokenText=String(numTokensInput??"").trim();
  if(!workspace)reasons.push({code:"no_workspace",message:"Load a workspace before starting reconstruction."});
  if(!exportRecord)reasons.push({code:"no_export",message:"Select an immutable export before starting reconstruction."});
  if(!health)reasons.push({code:"health_missing",message:"Check reconstruction services before starting."});
  else if(!health.configured){
    const message=!health.moge_configured?"MoGe is not configured.":!health.compiler_configured?"Reconstruction compiler is not configured.":health.error||"Reconstruction service unavailable.";
    reasons.push({code:"health_unavailable",message});
  }
  if(startActive)reasons.push({code:"start_active",message:"A reconstruction start request is already in progress."});
  if(exportCreationActive)reasons.push({code:"export_active",message:"Wait for export snapshot creation to finish."});
  if(exportRecord&&!exportRecord.archive_sha256)reasons.push({code:"missing_archive_sha",message:"The selected export does not have an archive SHA."});
  if(workspace&&!Number.isFinite(Number(workspace.workspace_revision)))reasons.push({code:"missing_revision",message:"Current workspace revision is unavailable."});
  if(!Number.isInteger(resolution)||resolution<1||resolution>9)reasons.push({code:"invalid_resolution",message:"Resolution level must be an integer from 1 to 9."});
  if(tokenText){
    const tokens=Number(tokenText);
    if(!/^[0-9]+$/.test(tokenText)||!Number.isInteger(tokens)||tokens<1)reasons.push({code:"invalid_tokens",message:"Number of tokens must be empty or a positive integer."});
  }
  return {
    ready:reasons.length===0,
    reasons,
    resolutionLevel:resolution,
    numTokens:tokenText?Number(tokenText):null,
    staleAllowed:exportRecord?.is_stale===true
  };
}

function readableKey(key){return String(key).replace(/_/g," ").replace(/\b\w/g,c=>c.toUpperCase())}
function primitive(value){return value===null||["string","number","boolean"].includes(typeof value)}

export function normalizeCompilationReport(report={}){
  const gates=report.quality_gates&&typeof report.quality_gates==="object"?report.quality_gates:{};
  return {
    passed:boolOrNull(report.passed),
    object_count:numberOrNull(report.object_count),
    universal_v3_invocations:numberOrNull(report.universal_v3_invocations),
    quality_gates:Object.entries(gates).map(([key,value])=>({key,label:readableKey(key),passed:Boolean(value)})),
    confidence_counts:report.confidence_counts&&typeof report.confidence_counts==="object"?{...report.confidence_counts}:null,
    placement_counts:report.placement_counts&&typeof report.placement_counts==="object"?{...report.placement_counts}:null,
    clean_scene_plan_sha256:stringOrNull(report.clean_scene_plan_sha256)
  };
}

export function normalizeMogeSummary(summary={}){
  const visible={};
  const raw={};
  for(const [key,value] of Object.entries(summary||{})){
    if(primitive(value))visible[key]=hidePathLikeValue(value);
    else if(Array.isArray(value)&&value.every(primitive))visible[key]=value.map(hidePathLikeValue);
    else if(value&&typeof value==="object"){
      const nested={};
      for(const [nestedKey,nestedValue] of Object.entries(value))if(primitive(nestedValue))nested[nestedKey]=hidePathLikeValue(nestedValue);
      if(Object.keys(nested).length)visible[key]=nested;
    }
    if(primitive(value))raw[key]=hidePathLikeValue(value);
  }
  return {visible,raw};
}

function preferJob(existing,incoming){
  if(!existing)return incoming;
  if(Number(incoming.job_version)>Number(existing.job_version))return incoming;
  if(Number(incoming.job_version)<Number(existing.job_version))return existing;
  if(isTerminalReconstructionJob(existing)&&isActiveReconstructionJob(incoming))return existing;
  return incoming;
}

export class ReconstructionWorkspaceState {
  constructor({setTimeoutFn=setTimeout,clearTimeoutFn=clearTimeout}={}){
    this.setTimeoutFn=setTimeoutFn;
    this.clearTimeoutFn=clearTimeoutFn;
    this.reset(null);
  }
  reset(workspaceId){
    this.stopPolling();
    this.workspaceGeneration=(this.workspaceGeneration||0)+1;
    this.workspaceId=workspaceId;
    this.health=null;
    this.healthLoading=false;
    this.healthError="";
    this.healthGeneration=(this.healthGeneration||0)+1;
    this.jobs=[];
    this.selectedJobId=null;
    this.jobsLoading=false;
    this.jobsError="";
    this.jobsGeneration=(this.jobsGeneration||0)+1;
    this.startOperation=null;
    this.nextOperationId=1;
    this.pollGeneration=(this.pollGeneration||0)+1;
    this.pollTimer=null;
    this.activePollContext=null;
    this.pollFailureCount=0;
    this.compilationReportByJob=new Map();
    this.mogeSummaryByJob=new Map();
    this.diagnosticGenerationByKey=new Map();
    this.diagnosticLoading=new Map();
    this.diagnosticError=new Map();
  }
  beginHealth(workspaceId){
    if(this.workspaceId!==workspaceId)this.reset(workspaceId);
    this.healthLoading=true;this.healthError="";this.healthGeneration+=1;
    return {workspaceId,generation:this.healthGeneration};
  }
  isCurrentHealth(context){return this.workspaceId===context.workspaceId&&this.healthGeneration===context.generation}
  setHealth(context,health){if(this.isCurrentHealth(context)){this.health=normalizeReconstructionHealth(health);this.healthLoading=false;return true}return false}
  failHealth(context,error){if(this.isCurrentHealth(context)){this.healthError=String(error?.message||error||"Service unavailable");this.healthLoading=false;return true}return false}
  beginJobs(workspaceId,{manual=false,poll=false,pollGeneration=null}={}){
    if(this.workspaceId!==workspaceId)this.reset(workspaceId);
    const kind=poll?"poll":"manual";
    if(manual){this.pollFailureCount=0;this.activePollContext=null}
    if(poll){
      const expectedPollGeneration=pollGeneration??this.pollGeneration;
      if(this.activePollContext||expectedPollGeneration!==this.pollGeneration)return null;
    }
    this.jobsLoading=true;this.jobsError="";this.jobsGeneration+=1;
    const context={
      workspaceId,
      workspaceGeneration:this.workspaceGeneration,
      jobsGeneration:this.jobsGeneration,
      pollGeneration:this.pollGeneration,
      kind
    };
    if(poll)this.activePollContext={...context};
    return context;
  }
  ownsPoll(context){
    return Boolean(context&&this.activePollContext&&this.activePollContext.workspaceId===context.workspaceId&&this.activePollContext.workspaceGeneration===context.workspaceGeneration&&this.activePollContext.jobsGeneration===context.jobsGeneration&&this.activePollContext.pollGeneration===context.pollGeneration);
  }
  isCurrentJobs(context){
    const base=this.workspaceId===context.workspaceId&&this.workspaceGeneration===context.workspaceGeneration&&this.jobsGeneration===context.jobsGeneration;
    if(!base)return false;
    return context.kind==="poll"?this.pollGeneration===context.pollGeneration:true;
  }
  setJobs(context,jobs){
    if(!this.isCurrentJobs(context))return false;
    const incoming=sortReconstructionJobsNewestFirst(jobs);
    const existing=new Map(this.jobs.map(job=>[job.job_id,job]));
    const merged=incoming.map(job=>preferJob(existing.get(job.job_id),job));
    this.jobs=sortReconstructionJobsNewestFirst(merged);
    if(this.selectedJobId&&!this.jobs.some(job=>job.job_id===this.selectedJobId))this.selectedJobId=null;
    if(!this.selectedJobId)this.selectedJobId=this.jobs[0]?.job_id??null;
    this.jobsLoading=false;this.pollFailureCount=0;
    return true;
  }
  failJobs(context,error,{poll=false}={}){
    if(!this.isCurrentJobs(context))return false;
    this.jobsError=String(error?.message||error||"Could not load reconstruction jobs.");
    this.jobsLoading=false;
    if(poll)this.pollFailureCount+=1;
    return true;
  }
  finishJobs(context){
    if(!context)return;
    if(context.kind==="poll"&&this.ownsPoll(context))this.activePollContext=null;
    if(this.workspaceId===context.workspaceId&&this.workspaceGeneration===context.workspaceGeneration&&this.jobsGeneration===context.jobsGeneration)this.jobsLoading=false;
  }
  upsertJob(job){
    const normalized=normalizeReconstructionJob(job);
    const byId=new Map(this.jobs.map(item=>[item.job_id,item]));
    byId.set(normalized.job_id,preferJob(byId.get(normalized.job_id),normalized));
    this.jobs=sortReconstructionJobsNewestFirst([...byId.values()]);
    this.selectedJobId=normalized.job_id;
    return normalized;
  }
  selectJob(jobId){if(this.jobs.some(job=>job.job_id===jobId))this.selectedJobId=jobId}
  selectedJob(){return this.jobs.find(job=>job.job_id===this.selectedJobId)||null}
  jobsForExport(exportId){return this.jobs.filter(job=>job.export_id===exportId)}
  hasActiveJobs(){return this.jobs.some(isActiveReconstructionJob)}
  beginStart(snapshot){if(this.startOperation)return false;this.startOperation={...snapshot};return true}
  finishStart(snapshot){if(this.startOperation?.operationId===snapshot.operationId)this.startOperation=null}
  isCurrentStart(snapshot){return Boolean(this.startOperation)&&this.startOperation.operationId===snapshot.operationId&&this.workspaceId===snapshot.workspaceId&&this.workspaceGeneration===snapshot.stateGeneration}
  startSnapshot(workspace,exportRecord,readiness){
    return createReconstructionRequestSnapshot({
      operationId:this.nextOperationId++,
      workspace,
      exportRecord,
      resolutionLevel:readiness.resolutionLevel,
      numTokens:readiness.numTokens,
      stateGeneration:this.workspaceGeneration
    });
  }
  stopPolling(){
    if(this.pollTimer){this.clearTimeoutFn(this.pollTimer);this.pollTimer=null}
  }
  schedulePoll(callback,delayMs){
    this.stopPolling();
    const context={workspaceId:this.workspaceId,pollGeneration:this.pollGeneration};
    this.pollTimer=this.setTimeoutFn(()=>callback(context),delayMs);
    return context;
  }
  invalidatePolling(){this.stopPolling();this.pollGeneration+=1;this.activePollContext=null}
  pollDelay(){
    if(!this.pollFailureCount)return 2000;
    return Math.min(15000,2000*(2**Math.min(this.pollFailureCount,3)));
  }
  diagnosticKey(job,url,type){return `${this.workspaceId}:${job.job_id}:${type}:${url}`}
  cachedDiagnostic(job,url,type){return type==="compilation"?this.compilationReportByJob.get(this.diagnosticKey(job,url,type)):this.mogeSummaryByJob.get(this.diagnosticKey(job,url,type))}
  beginDiagnostic(job,url,type){
    const key=this.diagnosticKey(job,url,type);
    const keyGeneration=(this.diagnosticGenerationByKey.get(key)||0)+1;
    this.diagnosticGenerationByKey.set(key,keyGeneration);
    const context={workspaceId:this.workspaceId,workspaceGeneration:this.workspaceGeneration,jobId:job.job_id,url,type,key,keyGeneration};
    this.diagnosticLoading.set(key,true);this.diagnosticError.delete(key);
    return context;
  }
  isCurrentDiagnostic(context){return this.workspaceId===context.workspaceId&&this.workspaceGeneration===context.workspaceGeneration&&this.diagnosticGenerationByKey.get(context.key)===context.keyGeneration}
  setDiagnostic(context,value){
    if(!this.isCurrentDiagnostic(context))return false;
    if(context.type==="compilation")this.compilationReportByJob.set(context.key,normalizeCompilationReport(value));
    else this.mogeSummaryByJob.set(context.key,normalizeMogeSummary(value));
    return true;
  }
  failDiagnostic(context,error){
    if(!this.isCurrentDiagnostic(context))return false;
    this.diagnosticError.set(context.key,String(error?.message||error||"Could not load diagnostic."));
    return true;
  }
  finishDiagnostic(context){
    if(this.isCurrentDiagnostic(context))this.diagnosticLoading.set(context.key,false);
  }
}
