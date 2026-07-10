import {manualMaskActive} from "./segment-state.js";

export function normalizeExportRecord(record={}){
  const masks=Array.isArray(record.masks)?record.masks:[];
  return {
    ...record,
    export_id:String(record.export_id||""),
    created_at:record.created_at||null,
    created_from_workspace_revision:Number(record.created_from_workspace_revision||0),
    published_workspace_revision:Number(record.published_workspace_revision||0),
    masks:masks.map(mask=>({
      ...mask,
      object_id:String(mask.object_id||""),
      object_version:Number(mask.object_version||0),
      semantic_label:String(mask.semantic_label||""),
      display_name:String(mask.display_name||""),
      source_kind:String(mask.source_kind||""),
      source_prompt_revision:Number(mask.source_prompt_revision||0),
      source_candidate_index:Number(mask.source_candidate_index||0),
      source_manual_revision:Number(mask.source_manual_revision||0),
      filename:String(mask.filename||""),
      mask_url:String(mask.mask_url||""),
      preview_url:String(mask.preview_url||""),
      mask_sha256:String(mask.mask_sha256||""),
      area_pixels:Number(mask.area_pixels||0),
      bbox_xyxy:Array.isArray(mask.bbox_xyxy)?mask.bbox_xyxy.map(Number):[0,0,0,0]
    })),
    metadata_url:String(record.metadata_url||""),
    quality_report_url:String(record.quality_report_url||""),
    quality_markdown_url:String(record.quality_markdown_url||""),
    combined_preview_url:String(record.combined_preview_url||""),
    archive_url:String(record.archive_url||""),
    archive_sha256:String(record.archive_sha256||""),
    warning_count:Number(record.warning_count||0),
    mask_count:Number(record.mask_count||masks.length||0),
    total_mask_area:Number(record.total_mask_area||0),
    is_stale:typeof record.is_stale==="boolean"?record.is_stale:null
  };
}

export function sortExportsNewestFirst(records=[]){
  return [...records].map(normalizeExportRecord).sort((a,b)=>{
    const byDate=Date.parse(b.created_at||"")-Date.parse(a.created_at||"");
    if(Number.isFinite(byDate)&&byDate!==0)return byDate;
    return b.published_workspace_revision-a.published_workspace_revision;
  });
}

export function createExportRequestSnapshot(workspace,operationId=1){
  return {
    operationId,
    workspaceId:workspace.workspace_id,
    expectedWorkspaceRevision:workspace.workspace_revision,
    expectedObjects:(workspace.objects||[]).map(object=>({
      object_id:object.object_id,
      expected_object_version:object.object_version
    }))
  };
}

export function exportRequestPayload(snapshot){
  return {
    expected_workspace_revision:snapshot.expectedWorkspaceRevision,
    expected_objects:snapshot.expectedObjects.map(item=>({...item})),
    include_previews:true
  };
}

export function effectiveMaskDescriptor(object){
  const draft=object?.sam_draft||{},manual=object?.manual_mask||{};
  if(manualMaskActive(object)){
    if(!manual.composite_mask_url)return {ok:false,code:"missing_manual_composite",message:`${object.display_name} has no saved manual composite artifact.`};
    if(manual.base_prompt_revision!==draft.prompt_revision||manual.base_candidate_index!==draft.selected_candidate_index){
      return {ok:false,code:"manual_base_mismatch",message:`${object.display_name} manual corrections are anchored to another candidate.`};
    }
    return {
      ok:true,
      sourceKind:"manual_composite",
      promptRevision:manual.base_prompt_revision,
      candidateIndex:manual.base_candidate_index,
      manualRevision:manual.manual_revision,
      url:manual.composite_mask_url
    };
  }
  if(draft.selected_candidate_index===null||draft.selected_candidate_index===undefined){
    return {ok:false,code:"missing_effective_mask",message:`${object.display_name} has no selected SAM candidate.`};
  }
  const candidate=(draft.candidates||[]).find(item=>item.candidate_index===draft.selected_candidate_index);
  if(!candidate||!candidate.mask_url){
    return {ok:false,code:"missing_effective_mask",message:`${object.display_name} selected SAM candidate is missing.`};
  }
  return {
    ok:true,
    sourceKind:"sam_candidate",
    promptRevision:draft.prompt_revision||0,
    candidateIndex:candidate.candidate_index,
    manualRevision:0,
    url:candidate.mask_url
  };
}

export function exportReadiness(workspace,{brushDirty=false,manualOperationActive=false,predictionActive=false,structuralBusy=false,exportActive=false}={}){
  const reasons=[];
  if(!workspace)reasons.push({code:"no_workspace",message:"Load or create a workspace before exporting."});
  if(workspace&&!(workspace.objects||[]).length)reasons.push({code:"no_objects",message:"Create at least one semantic object before exporting."});
  if(brushDirty)reasons.push({code:"dirty_brush",message:"Save or reset local brush edits before exporting."});
  if(manualOperationActive)reasons.push({code:"manual_operation_active",message:"A manual save or clear operation is in progress."});
  if(predictionActive)reasons.push({code:"prediction_active",message:"A SAM prediction is still running."});
  if(structuralBusy)reasons.push({code:"structural_operation_active",message:"An object operation is still running."});
  if(exportActive)reasons.push({code:"export_active",message:"Creating export snapshot..."});
  for(const object of workspace?.objects||[]){
    const descriptor=effectiveMaskDescriptor(object);
    if(!descriptor.ok)reasons.push({objectId:object.object_id,displayName:object.display_name,code:descriptor.code,message:descriptor.message});
  }
  return {ready:reasons.length===0,reasons};
}

export function exportRecordMatchesWorkspace(record,workspace){
  const normalized=normalizeExportRecord(record);
  if(!workspace||new Set((workspace.objects||[]).map(item=>item.object_id)).size!==normalized.masks.length)return false;
  const objects=new Map((workspace.objects||[]).map(item=>[item.object_id,item]));
  for(const mask of normalized.masks){
    const object=objects.get(mask.object_id);
    if(!object)return false;
    const descriptor=effectiveMaskDescriptor(object);
    if(!descriptor.ok)return false;
    if(object.object_version!==mask.object_version||object.semantic_label!==mask.semantic_label)return false;
    if(descriptor.sourceKind!==mask.source_kind||descriptor.promptRevision!==mask.source_prompt_revision||descriptor.candidateIndex!==mask.source_candidate_index||descriptor.manualRevision!==mask.source_manual_revision)return false;
  }
  return true;
}

export function normalizeQualityReport(report={}){
  return {
    ...report,
    masks:Array.isArray(report.masks)?report.masks:[],
    pairwise_overlaps:Array.isArray(report.pairwise_overlaps)?report.pairwise_overlaps:[],
    bbox_comparisons:Array.isArray(report.bbox_comparisons)?report.bbox_comparisons:[],
    summary:{
      ...(report.summary||{}),
      warnings:Array.isArray(report.summary?.warnings)?report.summary.warnings:[],
      mask_count:Number(report.summary?.mask_count||0)
    }
  };
}

export function staleLabel(record){
  return record?.is_stale===false?"Current":record?.is_stale===true?"Stale":"Unknown";
}

export function sourceKindLabel(value){
  return value==="manual_composite"?"Saved manual composite":value==="sam_candidate"?"SAM candidate":String(value||"Unknown");
}

export class ExportWorkspaceState {
  constructor(){this.reset(null)}
  reset(workspaceId){
    this.workspaceId=workspaceId;
    this.records=[];
    this.selectedExportId=null;
    this.loading=false;
    this.creating=false;
    this.detailsLoading=false;
    this.qualityLoading=false;
    this.qualityGeneration=0;
    this.error="";
    this.generation=(this.generation||0)+1;
    this.activeOperation=null;
    this.qualityReportByExport=new Map();
    this.nextOperationId=1;
  }
  beginList(workspaceId){
    if(this.workspaceId!==workspaceId)this.reset(workspaceId);
    this.loading=true;this.error="";
    this.generation+=1;
    return this.generation;
  }
  isCurrent(workspaceId,generation){return this.workspaceId===workspaceId&&this.generation===generation}
  setRecords(records){
    this.records=sortExportsNewestFirst(records);
    if(this.selectedExportId&&!this.records.some(item=>item.export_id===this.selectedExportId))this.selectedExportId=null;
    if(!this.selectedExportId)this.selectedExportId=this.records[0]?.export_id??null;
    this.loading=false;
  }
  selectedRecord(){return this.records.find(item=>item.export_id===this.selectedExportId)||null}
  select(exportId){if(this.records.some(item=>item.export_id===exportId))this.selectedExportId=exportId}
  beginCreate(workspace){
    if(this.activeOperation)return null;
    const snapshot=createExportRequestSnapshot(workspace,this.nextOperationId++);
    this.activeOperation=snapshot;
    this.creating=true;this.error="";
    return snapshot;
  }
  isCurrentOperation(snapshot,workspaceId){return Boolean(this.activeOperation)&&this.activeOperation.operationId===snapshot.operationId&&this.activeOperation.workspaceId===workspaceId&&this.workspaceId===workspaceId}
  finishCreate(snapshot){if(this.activeOperation?.operationId===snapshot.operationId){this.activeOperation=null;this.creating=false}}
  upsertRecord(record){
    const normalized=normalizeExportRecord(record);
    const remaining=this.records.filter(item=>item.export_id!==normalized.export_id);
    this.records=sortExportsNewestFirst([normalized,...remaining]);
    this.selectedExportId=normalized.export_id;
    return normalized;
  }
  qualityKey(record){return `${this.workspaceId}:${record.export_id}:${record.quality_report_url}`}
  cachedQuality(record){return this.qualityReportByExport.get(this.qualityKey(record))||null}
  setQuality(record,report){const normalized=normalizeQualityReport(report);this.qualityReportByExport.set(this.qualityKey(record),normalized);return normalized}
  beginQuality(record){
    this.qualityLoading=true;
    this.error="";
    this.qualityGeneration+=1;
    return {workspaceId:this.workspaceId,exportId:record.export_id,url:record.quality_report_url,generation:this.qualityGeneration};
  }
  isCurrentQuality(context){
    return this.workspaceId===context.workspaceId&&this.selectedExportId===context.exportId&&this.qualityGeneration===context.generation;
  }
  finishQuality(context){
    if(this.isCurrentQuality(context))this.qualityLoading=false;
  }
}
