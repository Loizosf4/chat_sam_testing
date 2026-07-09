import {ViewportTransform} from "/static/transform.js";
import {binaryFromImageData,colorForIndex,rgbaForMask} from "/static/mask-utils.js";
import {SegmentationWorkspaceClient} from "/static/segment-api.js";
import {
  ManualOperationRegistry,
  NO_SAM_MASK_MESSAGE,
  binaryMaskToPngBlob,
  brushBaseDescriptor,
  brushSaveState,
  brushSessionMatches,
  canStartManualOperation,
  createBrushSession,
  createManualOperationContext,
  manualConflictCompatibility,
  manualOperationMatchesCurrentState,
  markBrushEdited
} from "/static/segment-brush.js";
import {canvasEventPoint,canvasPointToImage,fitViewport} from "/static/segment-coordinates.js";
import {candidateListHtml,objectListHtml} from "/static/segment-components.js";
import {
  ExportWorkspaceState,
  exportReadiness,
  exportRequestPayload,
  sourceKindLabel,
  staleLabel
} from "/static/segment-exports.js";
import {
  CandidateMaskCache,
  ControllerRegistry,
  PredictionController,
  RenderGate,
  SegmentStore,
  WorkspaceLoadGate,
  canSelectCandidate,
  canSelectCandidateForObject,
  canUseSamPrompting,
  candidateCacheKey,
  effectiveMaskCacheKey,
  mergeCandidateSelectionResponse,
  mergeClearDraftResponse,
  mergeManualMaskResponse,
  mergePredictionResponse,
  manualMaskActive,
  predictionStatusVisible,
  selectedCandidate,
  semanticLabelFromDisplayName
} from "/static/segment-state.js";

const $=selector=>document.querySelector(selector);
const api=new SegmentationWorkspaceClient();
const store=new SegmentStore();
const view=new ViewportTransform();
const canvas=$("#segment-canvas");
const stage=$("#canvas-stage");
const ctx=canvas.getContext("2d");
const controllers=new ControllerRegistry();
const candidateImageCache=new CandidateMaskCache();
const loadGate=new WorkspaceLoadGate();
const refreshGate=new WorkspaceLoadGate();
const renderGate=new RenderGate();
const structuralBusyByObject=new Map();
const manualOps=new ManualOperationRegistry();
const exportState=new ExportWorkspaceState();
let sourceImage=null,sourceImageId=null,renderQueued=false,panPointer=null,brushSession=null,brushGeneration=0,brushPointer=null,brushCursor=null;

function absoluteUrl(path){return new URL(path,location.origin).href}
function esc(value){return String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c])}
function toast(message,error=false){const el=document.createElement("div");el.className=`toast${error?" error":""}`;el.textContent=message;$("#status-region").append(el);setTimeout(()=>el.remove(),4200)}
function setCanvasMessage(message){$("#canvas-message").textContent=message;$("#canvas-message").hidden=!message}
function currentObject(){return store.selected}
function currentPromptState(){return store.getPromptState(store.selectedObjectId)}
function selectedWorkspaceId(){return store.workspace?.workspace_id??null}
function selectedController(){return store.selectedObjectId&&selectedWorkspaceId()?controllers.get(selectedWorkspaceId(),store.selectedObjectId):null}
function brushDirty(){return Boolean(brushSession?.editor?.dirty)}
function selectedBrushSession(){return brushSession?.workspaceId===selectedWorkspaceId()&&brushSession?.objectId===store.selectedObjectId?brushSession:null}
function selectedManualOperation(){return store.selectedObjectId&&selectedWorkspaceId()?manualOps.get(selectedWorkspaceId(),store.selectedObjectId):null}
function selectedManualOperationActive(){return Boolean(selectedManualOperation())}
function anyManualOperationActive(){return manualOps.hasAny()}
function exportOperationActive(){return Boolean(exportState.activeOperation)}
function selectedBrushBusy(){const session=selectedBrushSession();return Boolean(session?.loading||session?.saving||session?.clearing||selectedManualOperationActive()||exportOperationActive())}
function clearObjectMaskCache(workspaceId,objectId){candidateImageCache.clearObject(workspaceId,objectId)}
function waitForManualOperation(){toast("Wait for the manual mask operation to finish.",true)}
function ensureNoSelectedManualOperation(){if(selectedManualOperationActive()){waitForManualOperation();return false}return true}
function ensureNoManualOperation(){if(anyManualOperationActive()){waitForManualOperation();return false}return true}
function waitForExportOperation(){toast("Wait for the export snapshot to finish.",true)}
function ensureNoExportOperation(){if(exportOperationActive()){waitForExportOperation();return false}return true}
function anyPredictionActive(){return [...controllers.controllers.values()].some(controller=>controller.running||controller.pending)||[...store.localPromptStateByObject.values()].some(state=>state.running||state.pending)}
function anyStructuralBusy(){return structuralBusyByObject.size>0}
function operationOwnsSession(operation,session){return Boolean(operation&&session&&session.workspaceId===operation.workspaceId&&session.objectId===operation.objectId&&session.generation===operation.brushGeneration&&session.identity===operation.brushIdentity)}
function setOwnedSessionBusy(operation,ownerSession,field,value){if(operationOwnsSession(operation,ownerSession))ownerSession[field]=value;if(operationOwnsSession(operation,brushSession))brushSession[field]=value}
function objectByOperation(operation){return store.workspace?.workspace_id===operation.workspaceId?store.workspace.objects.find(item=>item.object_id===operation.objectId):null}
function currentStateMatchesOperation(operation){const object=objectByOperation(operation);return manualOperationMatchesCurrentState(operation,{workspace:store.workspace,object,session:brushSession})}
function restoreBrushSession(session,object=currentObject()){if(session&&store.workspace?.workspace_id===session.workspaceId&&object?.object_id===session.objectId){brushSession=session;requestDraw();return true}return false}

function imageElement(url){return new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(new Error(`Could not load ${url}`));img.src=absoluteUrl(url)})}
async function maskBinary(candidate,draft,objectId,workspaceId){if(!candidate)return null;const key=candidateCacheKey({workspaceId,objectId,promptRevision:draft.prompt_revision,candidateIndex:candidate.candidate_index,maskUrl:candidate.mask_url});if(candidateImageCache.has(key))return candidateImageCache.get(key);const img=await imageElement(`${candidate.mask_url}?prompt_revision=${draft.prompt_revision}&candidate=${candidate.candidate_index}`);const c=document.createElement("canvas");c.width=view.imageWidth;c.height=view.imageHeight;const cctx=c.getContext("2d",{willReadFrequently:true});cctx.drawImage(img,0,0,c.width,c.height);const binary=binaryFromImageData(cctx.getImageData(0,0,c.width,c.height));candidateImageCache.set(key,binary);return binary}
async function maskBinaryFromUrl(url,key){if(!url)return null;if(candidateImageCache.has(key))return candidateImageCache.get(key);const img=await imageElement(url);const c=document.createElement("canvas");c.width=view.imageWidth;c.height=view.imageHeight;const cctx=c.getContext("2d",{willReadFrequently:true});cctx.drawImage(img,0,0,c.width,c.height);const binary=binaryFromImageData(cctx.getImageData(0,0,c.width,c.height));candidateImageCache.set(key,binary);return binary}

function disposeBrushSession(){brushGeneration+=1;brushSession=null;brushPointer=null;brushCursor=null;requestDraw()}
function brushDescriptorFor(object=currentObject()){return brushBaseDescriptor(store.workspace,object)}
function brushSessionStillCompatible(session=brushSession,object=currentObject()){
  if(!session||!store.workspace||!object)return false;
  const descriptor=brushDescriptorFor(object);
  return Boolean(descriptor.available)&&session.workspaceId===store.workspace.workspace_id&&session.objectId===object.object_id&&session.basePromptRevision===descriptor.basePromptRevision&&session.baseCandidateIndex===descriptor.baseCandidateIndex&&session.manualRevision===descriptor.manualRevision&&session.sourceMaskUrl===descriptor.sourceMaskUrl;
}
async function prepareBrushForSelection(){
  const workspace=store.workspace,object=currentObject();
  const generation=++brushGeneration;
  brushPointer=null;brushCursor=null;
  if(!workspace||!object){brushSession=null;renderAll();return}
  const descriptor=brushDescriptorFor(object);
  brushSession={...descriptor,workspaceId:workspace.workspace_id,objectId:object.object_id,sourceWidth:workspace.image_dimensions.width,sourceHeight:workspace.image_dimensions.height,generation,editor:null,editRevision:0,loading:Boolean(descriptor.available),saving:false,clearing:false,error:descriptor.error||""};
  renderAll();
  if(!descriptor.available)return;
  const key=effectiveMaskCacheKey({workspaceId:workspace.workspace_id,objectId:object.object_id,maskType:descriptor.maskType,promptRevision:descriptor.basePromptRevision,candidateIndex:descriptor.baseCandidateIndex,manualRevision:descriptor.manualRevision,maskUrl:descriptor.sourceMaskUrl});
  try{
    const mask=await maskBinaryFromUrl(descriptor.sourceMaskUrl,key);
    if(generation!==brushGeneration||store.workspace?.workspace_id!==workspace.workspace_id||store.selectedObjectId!==object.object_id)return;
    const current=currentObject();
    const session=createBrushSession({workspace:store.workspace,object:current,sourceWidth:workspace.image_dimensions.width,sourceHeight:workspace.image_dimensions.height,mask,generation});
    if(!brushSessionMatches(session,store.workspace,current,generation))return;
    brushSession=session;
  }catch(error){
    if(generation===brushGeneration&&store.selectedObjectId===object.object_id)brushSession={...brushSession,loading:false,error:error.message||"Could not load mask for manual corrections"};
  }
  renderAll();
}

async function loadSourceImage(workspace,generation){const image=await imageElement(workspace.source_image.url);if(!loadGate.isCurrent(generation))return false;sourceImage=image;sourceImageId=workspace.source_image.image_id;view.setImageSize(workspace.image_dimensions.width,workspace.image_dimensions.height).fit();resizeCanvas();return true}
async function prepareSam(generation=loadGate.generation,workspaceId=store.workspace?.workspace_id){if(!workspaceId||!loadGate.isCurrent(generation))return;store.setSamStatus("preparing","Preparing...");try{const result=await api.prepareSam(workspaceId);if(!loadGate.isCurrent(generation)||store.workspace?.workspace_id!==workspaceId)return;store.setSamStatus("ready",`Ready (${result.model_type} ${result.device})`)}catch(error){if(!loadGate.isCurrent(generation)||store.workspace?.workspace_id!==workspaceId)return;store.setSamStatus("failed",error.message);toast(error.message,true)}}
function resetTransientForWorkspaceChange(){controllers.clear();structuralBusyByObject.clear();candidateImageCache.clear();manualOps.invalidateAll();exportState.reset(null);refreshGate.begin();disposeBrushSession();renderGate.request();$("#mask-updating").hidden=true}
async function refreshExportHistory(workspaceId=selectedWorkspaceId(),selectId=exportState.selectedExportId){
  if(!workspaceId)return;
  const generation=exportState.beginList(workspaceId);
  renderExports();
  try{
    const records=await api.listExports(workspaceId);
    if(!exportState.isCurrent(workspaceId,generation)||store.workspace?.workspace_id!==workspaceId)return;
    exportState.setRecords(records);
    if(selectId&&exportState.records.some(item=>item.export_id===selectId))exportState.select(selectId);
  }catch(error){
    if(!exportState.isCurrent(workspaceId,generation))return;
    exportState.loading=false;
    exportState.error=error.message;
    toast(error.message,true);
  }finally{
    renderExports();
  }
}
function scheduleExportHistoryRefresh(workspaceId=selectedWorkspaceId(),selectId=exportState.selectedExportId){
  if(workspaceId)queueMicrotask(()=>{if(store.workspace?.workspace_id===workspaceId)refreshExportHistory(workspaceId,selectId)});
}
async function activateWorkspace(workspace,generation){if(!loadGate.isCurrent(generation))return false;resetTransientForWorkspaceChange();store.load(workspace);exportState.reset(workspace.workspace_id);history.replaceState(null,"",`/segment?workspace=${encodeURIComponent(workspace.workspace_id)}`);$("#workspace-id").value=workspace.workspace_id;$("#empty-state").hidden=true;$("#workspace-app").hidden=false;setCanvasMessage("");prepareBrushForSelection();requestDraw();refreshExportHistory(workspace.workspace_id);await prepareSam(generation,workspace.workspace_id);return loadGate.isCurrent(generation)}
async function loadWorkspace(workspaceId){const generation=loadGate.begin();try{store.loadingState="Loading";store.setError("");setCanvasMessage("Loading workspace...");const workspace=await api.getWorkspace(workspaceId);if(!loadGate.isCurrent(generation))return;if(!await loadSourceImage(workspace,generation))return;await activateWorkspace(workspace,generation)}catch(error){if(!loadGate.isCurrent(generation))return;setCanvasMessage("");toast(error.message,true);store.setError(error.message)}}
async function createWorkspace(file){const generation=loadGate.begin();try{const workspace=await api.createWorkspace(file);if(!loadGate.isCurrent(generation))return;if(!await loadSourceImage(workspace,generation))return;await activateWorkspace(workspace,generation)}catch(error){if(loadGate.isCurrent(generation))toast(error.message,true)}}

function updateWorkspace(workspace,preserveId=store.selectedObjectId){const previousWorkspaceId=store.workspace?.workspace_id??workspace.workspace_id;const previousIds=new Set(store.workspace?.objects.map(o=>o.object_id)||[]);const previousBrush=brushSession;store.load(workspace,preserveId);if(exportState.workspaceId!==workspace.workspace_id)exportState.reset(workspace.workspace_id);const currentIds=new Set(store.workspace?.objects.map(o=>o.object_id)||[]);for(const id of previousIds)if(!currentIds.has(id)){controllers.dispose(previousWorkspaceId,id);structuralBusyByObject.delete(id);manualOps.invalidateObject(previousWorkspaceId,id);clearObjectMaskCache(previousWorkspaceId,id);if(previousBrush?.objectId===id)disposeBrushSession()}if(!brushSessionStillCompatible(previousBrush,currentObject())){if(previousBrush?.editor?.dirty&&previousBrush.objectId===store.selectedObjectId)brushSession={...previousBrush,error:"Workspace state changed. Save is disabled until you reload or discard local brush edits."};else prepareBrushForSelection()}else brushSession=previousBrush;renderPredictionStatus();requestDraw();scheduleExportHistoryRefresh(workspace.workspace_id,exportState.selectedExportId)}
function selectedObjectColor(){const index=Math.max(0,(store.workspace?.objects||[]).findIndex(o=>o.object_id===store.selectedObjectId));return colorForIndex(index)}

function renderObjects(){if(!store.workspace)return;$("#object-list").innerHTML=objectListHtml(store.workspace.objects,store.selectedObjectId,store.localPromptStateByObject)}
function renderDetails(){
  const object=currentObject();
  const operationActive=selectedManualOperationActive();
  const busy=object?Boolean(structuralBusyByObject.get(object.object_id)||operationActive||exportOperationActive()):exportOperationActive();
  const prompt=currentPromptState();
  const session=selectedBrushSession();
  const dirty=brushDirty();
  const manual=object?.manual_mask||{};
  const manualActive=object?manualMaskActive(object):false;
  $("#rename-form").hidden=!object;
  $("#delete-object").disabled=!object||busy;
  $("#new-object").disabled=!store.workspace||exportOperationActive();
  const samReady=store.samStatus.state==="ready";
  const samPromptAllowed=canUseSamPrompting({object,promptState:prompt,brushDirty:dirty,structuralBusy:busy,samReady});
  document.querySelectorAll("[data-tool='positive'],[data-tool='negative']").forEach(button=>{button.disabled=!samPromptAllowed;button.title=!samReady?"SAM is not ready":manualActive?"Clear saved manual corrections before changing the SAM mask.":dirty?"Save or reset your brush edits before changing the SAM base.":""});
  $("#undo-point").disabled=!object||busy||manualActive||dirty||!prompt?.localPoints.length;
  $("#reset-draft").disabled=!object||busy;
  $("#undo-brush").disabled=!session?.editor?.undoStack.length||selectedBrushBusy();
  $("#redo-brush").disabled=!session?.editor?.redoStack.length||selectedBrushBusy();
  $("#reset-brush").disabled=!dirty||selectedBrushBusy();
  const anchorOk=brushSessionStillCompatible(session,object);
  $("#save-manual").disabled=!object||!session?.editor||!dirty||selectedBrushBusy()||busy||Boolean(selectedController()?.running||selectedController()?.pending)||!anchorOk;
  $("#clear-manual").disabled=!object||!manualActive||selectedBrushBusy()||busy;
  $("#brush-state").textContent=session?.loading?"Loading mask...":brushSaveState(session);
  if(!object){
    $("#candidate-list").innerHTML="<p class='muted-pad'>Select an object</p>";
    $("#manual-message").textContent="Select an object";
    return;
  }
  $("#display-name").value=object.display_name;
  $("#semantic-label").value=object.semantic_label;
  $("#point-count").textContent=prompt?.localPoints.length??0;
  $("#prompt-revision").textContent=object.sam_draft?.prompt_revision??0;
  $("#object-version").textContent=object.object_version;
  $("#manual-revision").textContent=manual.manual_revision??0;
  $("#manual-base-revision").textContent=manual.base_prompt_revision??"-";
  $("#manual-base-candidate").textContent=manual.base_candidate_index!==null&&manual.base_candidate_index!==undefined?Number(manual.base_candidate_index)+1:"-";
  $("#manual-area").textContent=manual.area_pixels??0;
  $("#manual-bbox").textContent=`[${(manual.bbox_xyxy||[0,0,0,0]).join(", ")}]`;
  $("#manual-message").textContent=session?.error||(
    manualActive?"Saved manual corrections are active. Clear them before changing the SAM mask.":"No saved manual corrections"
  );
  $("#candidate-list").innerHTML=candidateListHtml(object,object.sam_draft?.selected_candidate_index);
  for(const button of document.querySelectorAll("[data-candidate-index]")){
    button.disabled=!canSelectCandidateForObject({object,promptState:prompt,brushDirty:dirty,structuralBusy:busy});
    if(button.disabled)button.title=manualActive?"Clear saved manual corrections before changing the SAM mask.":dirty?"Save or reset your brush edits before changing the SAM base.":"Wait for the current prediction.";
  }
}
function renderSamStatus(){const el=$("#sam-status");el.dataset.state=store.samStatus.state;el.querySelector("output").textContent=store.samStatus.message;$("#retry-sam").hidden=!store.workspace}
function renderPredictionStatus(){const controller=selectedController();const state=currentPromptState();$("#mask-updating").hidden=!predictionStatusVisible({controller,promptState:state})}
function formatDate(value){const date=new Date(value);return Number.isNaN(date.getTime())?"-":date.toLocaleString()}
function formatNumber(value){return Number(value||0).toLocaleString()}
function shortHash(value){return value?`${value.slice(0,12)}...${value.slice(-8)}`:"-"}
function badgeHtml(record){const label=staleLabel(record);return `<span class="badge ${label.toLowerCase()}">${label}</span>`}
function exportReadinessState(){
  return exportReadiness(store.workspace,{
    brushDirty:brushDirty(),
    manualOperationActive:anyManualOperationActive(),
    predictionActive:anyPredictionActive(),
    structuralBusy:anyStructuralBusy(),
    exportActive:exportOperationActive()
  });
}
function renderExports(){
  const section=$("#export-section");
  if(!section)return;
  section.setAttribute("aria-busy",String(exportState.loading||exportState.creating||exportState.qualityLoading));
  const readiness=exportReadinessState();
  const create=$("#create-export");
  create.disabled=!readiness.ready||exportState.creating;
  create.textContent=exportState.creating?"Creating export snapshot...":"Create export snapshot";
  const status=$("#export-status");
  status.className=`export-status ${readiness.ready?"is-ready":"is-blocked"}`;
  status.textContent=exportState.creating?"Creating export snapshot...":readiness.ready?"Ready to export":"Export blocked";
  $("#export-readiness").innerHTML=readiness.ready?"":readiness.reasons.map(reason=>`<p class="export-reason">${esc(reason.message)}</p>`).join("");
  $("#refresh-exports").disabled=!store.workspace||exportState.loading||exportState.creating;
  const history=$("#export-history");
  if(!store.workspace)history.innerHTML="<p class='muted-pad'>Load a workspace</p>";
  else if(exportState.loading)history.innerHTML="<p class='muted-pad'>Loading exports...</p>";
  else if(exportState.error)history.innerHTML=`<p class="export-reason">${esc(exportState.error)}</p>`;
  else if(!exportState.records.length)history.innerHTML="<p class='muted-pad'>No exports yet</p>";
  else history.innerHTML=exportState.records.map((record,index)=>`<button type="button" class="export-row ${record.export_id===exportState.selectedExportId?"is-selected":""}" data-export-id="${esc(record.export_id)}">
    <span><strong>Export ${exportState.records.length-index}</strong><small>${esc(formatDate(record.created_at))}</small><small>${record.mask_count} masks - ${record.warning_count} warnings</small></span>
    <span>${badgeHtml(record)}<small>rev ${record.created_from_workspace_revision} -> ${record.published_workspace_revision}</small></span>
  </button>`).join("");
  renderExportDetails();
}
function renderExportDetails(){
  const container=$("#export-details");
  const record=exportState.selectedRecord();
  if(!record){container.innerHTML="";return}
  const quality=exportState.cachedQuality(record);
  container.innerHTML=`
    <section class="export-details">
      <dl class="export-summary">
        <div><dt>Export ID</dt><dd class="hash-text" title="${esc(record.export_id)}">${esc(record.export_id)}</dd></div>
        <div><dt>Status</dt><dd>${badgeHtml(record)}</dd></div>
        <div><dt>Created</dt><dd>${esc(formatDate(record.created_at))}</dd></div>
        <div><dt>Masks</dt><dd>${record.mask_count}</dd></div>
        <div><dt>Warnings</dt><dd>${record.warning_count}</dd></div>
        <div><dt>Total area</dt><dd>${formatNumber(record.total_mask_area)} px</dd></div>
        <div><dt>Source rev</dt><dd>${record.created_from_workspace_revision}</dd></div>
        <div><dt>Published rev</dt><dd>${record.published_workspace_revision}</dd></div>
        <div><dt>Archive SHA</dt><dd class="hash-text" title="${esc(record.archive_sha256)}">${esc(shortHash(record.archive_sha256))}</dd></div>
      </dl>
      <div class="artifact-actions">
        <a href="${esc(record.archive_url)}" download>Download ZIP</a>
        <a href="${esc(record.metadata_url)}" target="_blank" rel="noopener">Open metadata JSON</a>
        <a href="${esc(record.quality_report_url)}" target="_blank" rel="noopener">Open quality JSON</a>
        <a href="${esc(record.quality_markdown_url)}" target="_blank" rel="noopener">Open quality Markdown</a>
        <a href="${esc(record.combined_preview_url)}" target="_blank" rel="noopener">Open combined preview</a>
      </div>
      <div class="export-preview">
        <strong>Combined preview</strong>
        <a href="${esc(record.combined_preview_url)}" target="_blank" rel="noopener"><img src="${esc(record.combined_preview_url)}" alt="Combined mask overlay preview for export ${esc(record.export_id)}" loading="lazy"></a>
      </div>
      <details class="quality-block" id="quality-details" ${quality?"open":""}>
        <summary>Quality report ${exportState.qualityLoading?"(loading...)":""}</summary>
        <div id="quality-content">${quality?qualityHtml(quality):"<p class='export-muted'>Open to load quality diagnostics.</p>"}</div>
      </details>
      <div class="mask-list">
        <strong>Mask snapshots</strong>
        ${record.masks.map(mask=>maskSnapshotHtml(mask)).join("")}
      </div>
    </section>`;
}
function maskSnapshotHtml(mask){
  return `<article class="mask-snapshot">
    <strong>${esc(mask.display_name)}</strong>
    <div class="mask-grid">
      <span><b>Semantic label</b>${esc(mask.semantic_label)}</span>
      <span><b>Stable object ID</b><span class="hash-text">${esc(mask.object_id)}</span></span>
      <span><b>Object version</b>${mask.object_version}</span>
      <span><b>Source</b>${esc(sourceKindLabel(mask.source_kind))}</span>
      <span><b>SAM prompt</b>${mask.source_prompt_revision}</span>
      <span><b>Candidate</b>Candidate ${mask.source_candidate_index+1}</span>
      <span><b>Manual rev</b>${mask.source_manual_revision}</span>
      <span><b>Area</b>${formatNumber(mask.area_pixels)} px</span>
      <span><b>Bounds</b>[${mask.bbox_xyxy.join(", ")}]</span>
      <span><b>Filename</b>${esc(mask.filename)}</span>
      <span><b>Mask SHA</b><span class="hash-text" title="${esc(mask.mask_sha256)}">${esc(shortHash(mask.mask_sha256))}</span></span>
    </div>
    <div class="artifact-actions"><a href="${esc(mask.mask_url)}" target="_blank" rel="noopener">Open mask</a><a href="${esc(mask.preview_url)}" target="_blank" rel="noopener">Open overlay</a></div>
  </article>`;
}
function qualityHtml(report){
  const warnings=report.summary?.warnings||[];
  const overlaps=(report.pairwise_overlaps||[]).filter(item=>Number(item.overlap_pixels)>0);
  const bbox=(report.bbox_comparisons||[]).filter(item=>Number(item.bbox_iou)>=.9);
  return `<div class="quality-block">
    <strong>${warnings.length?`${warnings.length} quality warning${warnings.length===1?"":"s"}`:"No quality warnings"}</strong>
    <ul class="quality-list">${warnings.length?warnings.map(w=>`<li><b>${esc(w.severity||"warning")}</b> ${esc(w.label||w.mask_id||"mask")} - ${esc(w.message||"")}</li>`).join(""):"<li>No quality warnings</li>"}</ul>
    <strong>Per-mask metrics</strong>
    <ul class="quality-list">${(report.masks||[]).map(m=>`<li>${esc(m.label)}: ${formatNumber(m.area)} px, ${m.percent_image_area}% image, ${m.connected_component_count} components, largest ${formatNumber(m.largest_component_area)}, small ${m.small_component_count}, border ${m.touches_image_border?"yes":"no"}, bbox [${(m.bbox||[]).join(", ")}]</li>`).join("")}</ul>
    <strong>Overlaps</strong>
    <ul class="quality-list">${overlaps.length?overlaps.map(o=>`<li>${esc(o.label_a||o.mask_a)} / ${esc(o.label_b||o.mask_b)}: ${formatNumber(o.overlap_pixels)} px, ${o.overlap_percent_of_smaller_mask}% smaller, ${o.overlap_percent_of_union}% union</li>`).join(""):"<li>No overlapping mask pairs</li>"}</ul>
    <details><summary>Bounding-box diagnostics</summary><ul class="quality-list">${bbox.length?bbox.map(b=>`<li>${esc(b.label_a||b.mask_a)} / ${esc(b.label_b||b.mask_b)}: IoU ${b.bbox_iou}</li>`).join(""):"<li>No high-IoU bounding boxes</li>"}</ul></details>
  </div>`;
}
function renderAll(){renderObjects();renderDetails();renderSamStatus();renderPredictionStatus();renderExports();requestDraw()}

async function createExportSnapshot(){
  const readiness=exportReadinessState();
  if(!readiness.ready){toast(readiness.reasons[0]?.message||"Export is not ready.",true);renderExports();return}
  const snapshot=exportState.beginCreate(store.workspace);
  if(!snapshot){waitForExportOperation();return}
  const payload=exportRequestPayload(snapshot);
  renderAll();
  try{
    const record=await api.createExport({
      workspaceId:snapshot.workspaceId,
      expectedWorkspaceRevision:payload.expected_workspace_revision,
      expectedObjects:payload.expected_objects
    });
    if(!exportState.isCurrentOperation(snapshot,snapshot.workspaceId)||store.workspace?.workspace_id!==snapshot.workspaceId)return;
    exportState.upsertRecord(record);
    if(store.workspace)store.load({...store.workspace,workspace_revision:record.published_workspace_revision},store.selectedObjectId);
    await refreshActiveWorkspace(snapshot.workspaceId,store.selectedObjectId);
    await refreshExportHistory(snapshot.workspaceId,record.export_id);
    exportState.select(record.export_id);
    toast("Export snapshot created");
  }catch(error){
    if(error.status===409){
      await refreshActiveWorkspace(snapshot.workspaceId,store.selectedObjectId,"Export was stale. Reloaded workspace state.");
      await refreshExportHistory(snapshot.workspaceId);
    }else toast(error.message,true);
  }finally{
    exportState.finishCreate(snapshot);
    renderAll();
  }
}

async function loadSelectedQualityReport(){
  const record=exportState.selectedRecord();
  if(!record||exportState.cachedQuality(record)||exportState.qualityLoading)return;
  const context=exportState.beginQuality(record);
  renderExports();
  try{
    const report=await api.getJsonArtifact(context.url);
    if(!exportState.isCurrentQuality(context)||store.workspace?.workspace_id!==context.workspaceId)return;
    exportState.setQuality(record,report);
  }catch(error){
    if(exportState.isCurrentQuality(context)){exportState.error=error.message;toast(error.message,true)}
  }finally{
    exportState.finishQuality(context);
    renderExports();
  }
}

function drawPoint(point){
  const canvasPoint=view.imageToCanvas(point.x,point.y);
  const positive=point.label===1;
  ctx.save();
  ctx.lineWidth=2;
  ctx.fillStyle=positive?"#153d25":"#411b1f";
  ctx.strokeStyle=positive?"#55e08b":"#ff7474";
  ctx.beginPath();
  ctx.arc(canvasPoint.x,canvasPoint.y,7,0,Math.PI*2);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(canvasPoint.x-4,canvasPoint.y);
  ctx.lineTo(canvasPoint.x+4,canvasPoint.y);
  if(positive){ctx.moveTo(canvasPoint.x,canvasPoint.y-4);ctx.lineTo(canvasPoint.x,canvasPoint.y+4)}
  ctx.stroke();
  ctx.restore();
}

function drawBrushCursor(){
  const radius=Number($("#brush-size").value)*view.scale;
  ctx.save();
  ctx.beginPath();
  ctx.arc(brushCursor.x,brushCursor.y,radius,0,Math.PI*2);
  ctx.strokeStyle=store.activeTool==="manual-add"?"#9ff0c1":"#ffb2b8";
  ctx.lineWidth=1.5;
  ctx.setLineDash([5,4]);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle=ctx.strokeStyle;
  ctx.font="16px system-ui";
  ctx.textAlign="center";
  ctx.textBaseline="middle";
  ctx.fillText(store.activeTool==="manual-add"?"+":"-",brushCursor.x,brushCursor.y);
  ctx.restore();
}

function renderSnapshotState(){
  const object=currentObject(),draft=object?.sam_draft||{},candidate=object?selectedCandidate(object):null;
  const descriptor=object?brushDescriptorFor(object):null;
  const session=selectedBrushSession();
  return {
    workspaceId:store.workspace?.workspace_id??null,
    selectedObjectId:object?.object_id??null,
    promptRevision:draft.prompt_revision||0,
    selectedCandidateIndex:draft.selected_candidate_index??null,
    maskUrl:descriptor?.sourceMaskUrl??candidate?.mask_url??null,
    manualRevision:object?.manual_mask?.manual_revision??0,
    brushGeneration:session?.generation??brushGeneration,
    brushEditRevision:session?.editRevision??0,
    effectiveMaskIdentity:session?.identity??(descriptor?.sourceMaskUrl??candidate?.mask_url??null),
    viewport:{zoom:view.zoom,panX:view.panX,panY:view.panY,width:view.viewportWidth,height:view.viewportHeight,scale:view.scale},
    sourceImageId
  };
}

function paintBase(width,height){
  ctx.setTransform(Math.max(1,window.devicePixelRatio||1),0,0,Math.max(1,window.devicePixelRatio||1),0,0);
  ctx.clearRect(0,0,width,height);
  if(!sourceImage||!store.workspace)return false;
  const origin=view.origin,scale=view.scale;
  ctx.drawImage(sourceImage,origin.x,origin.y,view.imageWidth*scale,view.imageHeight*scale);
  return true;
}

async function draw(){
  renderQueued=false;
  const width=stage.clientWidth,height=stage.clientHeight;
  if(!width||!height)return;
  const dpr=Math.max(1,window.devicePixelRatio||1);
  if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)){
    canvas.width=Math.round(width*dpr);
    canvas.height=Math.round(height*dpr);
    view.setViewport(width,height,dpr);
  }
  const snapshot=renderGate.snapshot(renderSnapshotState());
  const object=currentObject(),draft=object?.sam_draft||{},candidate=object?selectedCandidate(object):null,session=selectedBrushSession();
  let mask=session?.editor?.mask??null;
  if(!mask&&object){
    const descriptor=brushDescriptorFor(object);
    if(descriptor.available){
      const key=effectiveMaskCacheKey({workspaceId:store.workspace.workspace_id,objectId:object.object_id,maskType:descriptor.maskType,promptRevision:descriptor.basePromptRevision,candidateIndex:descriptor.baseCandidateIndex,manualRevision:descriptor.manualRevision,maskUrl:descriptor.sourceMaskUrl});
      mask=await maskBinaryFromUrl(descriptor.sourceMaskUrl,key).catch(error=>{if(renderGate.isCurrent(snapshot,renderSnapshotState()))toast(`Mask artifact failed to load: ${error.message}`,true);return null});
      if(!renderGate.isCurrent(snapshot,renderSnapshotState()))return;
    }else if(candidate){
      mask=await maskBinary(candidate,draft,object.object_id,store.workspace.workspace_id).catch(error=>{if(renderGate.isCurrent(snapshot,renderSnapshotState()))toast(`Candidate artifact failed to load: ${error.message}`,true);return null});
      if(!renderGate.isCurrent(snapshot,renderSnapshotState()))return;
    }
  }
  if(!paintBase(width,height))return;
  if(object&&mask){
    const origin=view.origin,scale=view.scale;
    const overlay=document.createElement("canvas");
    overlay.width=view.imageWidth;
    overlay.height=view.imageHeight;
    overlay.getContext("2d").putImageData(rgbaForMask(mask,view.imageWidth,view.imageHeight,selectedObjectColor(),Number($("#overlay-opacity").value),true),0,0);
    ctx.imageSmoothingEnabled=false;
    ctx.drawImage(overlay,origin.x,origin.y,view.imageWidth*scale,view.imageHeight*scale);
  }
  if(object)for(const point of currentPromptState()?.localPoints||draft.points||[])drawPoint(point);
  if(brushCursor&&(store.activeTool==="manual-add"||store.activeTool==="manual-remove"))drawBrushCursor();
}
function requestDraw(){renderGate.request();if(!renderQueued){renderQueued=true;requestAnimationFrame(draw)}}
function resizeCanvas(){const rect=stage.getBoundingClientRect();fitViewport(view,rect.width,rect.height,window.devicePixelRatio||1);requestDraw();updateZoom()}
function updateZoom(){$("#zoom-level").textContent=`${Math.round(view.zoom*100)}%`}

function syncPromptStateFromController({workspaceId,objectId,running,pending,invalidated}){
  if(store.workspace?.workspace_id!==workspaceId)return;
  const state=store.getPromptState(objectId);
  if(state){state.running=running;state.pending=pending;state.invalidated=invalidated}
  renderPredictionStatus();
  renderDetails();
}
function getController(objectId){
  const workspaceId=store.workspace?.workspace_id;
  if(!workspaceId)return null;
  if(!controllers.get(workspaceId,objectId)){
    controllers.set(workspaceId,objectId,new PredictionController({
      workspaceId,
      objectId,
      isActive:id=>store.workspace?.workspace_id===workspaceId&&store.workspace?.objects.some(o=>o.object_id===id),
      submit:snapshot=>api.predict(workspaceId,objectId,{
        prompt_revision:snapshot.revision,
        points:snapshot.points,
        box:snapshot.box,
        base_prompt_revision:snapshot.basePromptRevision,
        base_candidate_index:snapshot.baseCandidateIndex,
        multimask_output:null
      }),
      apply:(response,snapshot)=>{
        if(!store.workspace||store.workspace.workspace_id!==workspaceId||!store.workspace.objects.some(o=>o.object_id===response.object_id))return;
        clearObjectMaskCache(workspaceId,response.object_id);
        if(brushSession?.objectId===response.object_id)disposeBrushSession();
        updateWorkspace(mergePredictionResponse(store.workspace,response),store.selectedObjectId);
        const state=store.getPromptState(response.object_id);
        state.latestAppliedRevision=response.prompt_revision;
        state.persistedRevision=response.prompt_revision;
        state.localPoints=[...snapshot.points];
        prepareBrushForSelection();
        renderPredictionStatus();
      },
      onError:(error,snapshot)=>{
        if(store.workspace?.workspace_id!==workspaceId)return;
        const state=store.getPromptState(snapshot.objectId);
        if(error.status===409&&state?.nextRevision>snapshot.revision+1)return;
        if(error.status===409)reloadAfterConflict("Prediction was stale. Reloaded workspace state.");
        else toast(error.message,true);
      },
      onStateChange:syncPromptStateFromController
    }));
  }
  return controllers.get(workspaceId,objectId);
}

function submitPrompt(objectId,revision,points,box){
  const object=store.workspace.objects.find(o=>o.object_id===objectId);
  if(exportOperationActive()){waitForExportOperation();return}
  if(manualOps.has(store.workspace.workspace_id,objectId)){waitForManualOperation();return}
  if(manualMaskActive(object)){toast("Clear saved manual corrections before changing the SAM mask.",true);return}
  if(brushDirty()){toast("Save or reset your brush edits before changing the SAM base.",true);return}
  const draft=object?.sam_draft||{};
  const hasBase=Number.isInteger(draft.prompt_revision)&&draft.prompt_revision>0&&draft.selected_candidate_index!==null&&draft.selected_candidate_index!==undefined;
  const controller=getController(objectId);
  if(!controller)return;
  const promptState=store.getPromptState(objectId);
  controller.enqueue({
    objectId,
    revision,
    points:points.map(p=>({...p})),
    box,
    basePromptRevision:hasBase?draft.prompt_revision:null,
    baseCandidateIndex:hasBase?draft.selected_candidate_index:null
  });
  promptState.running=controller.running;promptState.pending=Boolean(controller.pending);
  renderPredictionStatus();
}

function disposeController(objectId){const workspaceId=store.workspace?.workspace_id;if(workspaceId)controllers.dispose(workspaceId,objectId);const state=store.getPromptState(objectId);if(state){state.running=false;state.pending=false;state.invalidated=false}renderPredictionStatus()}

async function clearDraft(objectId){
  const object=store.workspace.objects.find(o=>o.object_id===objectId);
  if(!object||structuralBusyByObject.get(objectId))return;
  if(exportOperationActive()){waitForExportOperation();return}
  if(manualOps.has(store.workspace.workspace_id,objectId)){waitForManualOperation();return}
  if((manualMaskActive(object)||brushDirty())&&!confirm("Resetting the SAM draft will remove SAM points, SAM candidates, saved manual corrections, and unsaved brush edits. Continue?"))return;
  const previousBrush=brushSession;
  structuralBusyByObject.set(objectId,"reset");
  disposeController(objectId);
  disposeBrushSession();
  renderAll();
  try{
    const response=await api.clearDraft(store.workspace,object);
    updateWorkspace(mergeClearDraftResponse(store.workspace,response),objectId);
    store.resetPromptState(objectId);
    clearObjectMaskCache(store.workspace.workspace_id,objectId);
    toast("SAM draft reset");
  }catch(error){
    const latestObject=store.workspace?.objects.find(item=>item.object_id===objectId);
    if(error.status===409)await reloadAfterConflict("Workspace changed. Reloaded; retry reset.");
    else{restoreBrushSession(previousBrush,latestObject);toast(error.message,true)}
  }
  finally{structuralBusyByObject.delete(objectId);renderAll()}
}

async function refreshActiveWorkspace(workspaceId=selectedWorkspaceId(),preserveId=store.selectedObjectId,message=""){
  if(!workspaceId)return null;
  const generation=refreshGate.begin();
  const workspace=await api.getWorkspace(workspaceId);
  if(!refreshGate.isCurrent(generation)||store.workspace?.workspace_id!==workspaceId)return null;
  updateWorkspace(workspace,preserveId);
  if(message)toast(message,true);
  return workspace;
}

async function reloadAfterConflict(message){await refreshActiveWorkspace(selectedWorkspaceId(),store.selectedObjectId,message)}

async function saveManualMask(){
  const object=currentObject(),session=selectedBrushSession();
  if(exportOperationActive()){waitForExportOperation();return}
  const predictionActive=Boolean(selectedController()?.running||selectedController()?.pending);
  const structuralBusy=Boolean(object&&structuralBusyByObject.get(object.object_id));
  if(!canStartManualOperation({workspace:store.workspace,object,session,operationActive:selectedManualOperationActive(),predictionActive,structuralBusy,type:"save"}))return;
  if(!brushSessionStillCompatible(session,object)){toast("Brush base changed. Reload or reset local edits before saving.",true);return}
  const operation=manualOps.begin(createManualOperationContext({type:"save",workspace:store.workspace,object,session}));
  if(!operation){waitForManualOperation();return}
  const ownerSession=session;
  setOwnedSessionBusy(operation,ownerSession,"saving",true);
  ownerSession.error="";
  renderAll();
  try{
    const blob=await binaryMaskToPngBlob(operation.editedMask,session.sourceWidth,session.sourceHeight);
    if(!manualOps.isCurrent(operation))return;
    const response=await api.saveManualMask({
      workspaceId:operation.workspaceId,
      objectId:operation.objectId,
      expectedWorkspaceRevision:operation.expectedWorkspaceRevision,
      expectedObjectVersion:operation.expectedObjectVersion,
      expectedManualRevision:operation.expectedManualRevision,
      basePromptRevision:operation.basePromptRevision,
      baseCandidateIndex:operation.baseCandidateIndex,
      blob
    });
    if(!manualOps.isCurrent(operation))return;
    if(!currentStateMatchesOperation(operation)){
      manualOps.finish(operation);
      setOwnedSessionBusy(operation,ownerSession,"saving",false);
      await refreshActiveWorkspace(operation.workspaceId,operation.objectId,"Manual save completed, but the local view had moved on. Refreshed workspace state.");
      return;
    }
    const edited=new Uint8Array(operation.editedMask);
    const merged=mergeManualMaskResponse(store.workspace,response);
    updateWorkspace(merged,operation.objectId);
    const updated=currentObject();
    const updatedSession=createBrushSession({workspace:store.workspace,object:updated,sourceWidth:ownerSession.sourceWidth,sourceHeight:ownerSession.sourceHeight,mask:edited,generation:++brushGeneration});
    brushSession=updatedSession;
    const key=effectiveMaskCacheKey({workspaceId:store.workspace.workspace_id,objectId:operation.objectId,maskType:"manual-composite",promptRevision:updatedSession.basePromptRevision,candidateIndex:updatedSession.baseCandidateIndex,manualRevision:updatedSession.manualRevision,maskUrl:updatedSession.sourceMaskUrl});
    candidateImageCache.set(key,new Uint8Array(edited));
    toast("Manual corrections saved");
  }catch(error){
    if(error.status===409){
      const latest=await refreshActiveWorkspace(operation.workspaceId,operation.objectId);
      const compatibility=manualConflictCompatibility(operation,latest);
      if(restoreBrushSession(ownerSession,compatibility.object||objectByOperation(operation))){
        ownerSession.saving=false;
        ownerSession.error=compatibility.compatible?"Workspace versions changed. Local edits preserved; retry save.":"Manual save conflict. The SAM base or manual revision changed; reload or reset local brush edits.";
      }
      toast(compatibility.compatible?"Workspace versions changed. Local edits preserved; retry save.":"Manual corrections conflicted. Review the current workspace before retrying.",true);
    }else{
      if(operationOwnsSession(operation,ownerSession))ownerSession.error=error.message;
      toast(error.message,true);
    }
  }finally{
    setOwnedSessionBusy(operation,ownerSession,"saving",false);
    manualOps.finish(operation);
    renderAll();
  }
}

async function clearManualMask(){
  const object=currentObject(),session=selectedBrushSession();
  if(exportOperationActive()){waitForExportOperation();return}
  const structuralBusy=Boolean(object&&structuralBusyByObject.get(object.object_id));
  if(!canStartManualOperation({workspace:store.workspace,object,session,operationActive:selectedManualOperationActive(),structuralBusy,type:"clear"}))return;
  if(session?.editor?.dirty&&!confirm("Clear saved corrections and discard unsaved brush edits?"))return;
  const operation=manualOps.begin(createManualOperationContext({type:"clear",workspace:store.workspace,object,session}));
  if(!operation){waitForManualOperation();return}
  const ownerSession=session;
  setOwnedSessionBusy(operation,ownerSession,"clearing",true);
  renderAll();
  try{
    const response=await api.clearManualMask({
      workspaceId:operation.workspaceId,
      objectId:operation.objectId,
      expectedWorkspaceRevision:operation.expectedWorkspaceRevision,
      expectedObjectVersion:operation.expectedObjectVersion,
      expectedManualRevision:operation.expectedManualRevision
    });
    if(!manualOps.isCurrent(operation))return;
    if(!currentStateMatchesOperation(operation)){
      manualOps.finish(operation);
      setOwnedSessionBusy(operation,ownerSession,"clearing",false);
      await refreshActiveWorkspace(operation.workspaceId,operation.objectId,"Manual clear completed, but the local view had moved on. Refreshed workspace state.");
      return;
    }
    const merged=mergeManualMaskResponse(store.workspace,response);
    clearObjectMaskCache(operation.workspaceId,operation.objectId);
    updateWorkspace(merged,operation.objectId);
    await prepareBrushForSelection();
    toast("Manual corrections cleared");
  }catch(error){
    if(error.status===409){
      const latest=await refreshActiveWorkspace(operation.workspaceId,operation.objectId);
      const compatibility=manualConflictCompatibility(operation,latest);
      if(restoreBrushSession(ownerSession,compatibility.object||objectByOperation(operation))){
        ownerSession.clearing=false;
        ownerSession.error=compatibility.compatible?"Workspace versions changed. Retry clear.":"Manual clear conflict. The SAM base or manual revision changed.";
      }
      toast(compatibility.compatible?"Workspace versions changed. Retry clear.":"Manual corrections changed. Reloaded; retry clear.",true);
    }else{
      if(operationOwnsSession(operation,ownerSession))ownerSession.error=error.message;
      toast(error.message,true);
    }
  }finally{
    setOwnedSessionBusy(operation,ownerSession,"clearing",false);
    manualOps.finish(operation);
    renderAll();
  }
}

function confirmDiscardBrushEdits(message){if(anyManualOperationActive()){waitForManualOperation();return false}if(exportOperationActive()){waitForExportOperation();return false}return !brushDirty()||confirm(message)}

function updateBrushDirty(){markBrushEdited(brushSession);renderDetails();requestDraw()}
function beginBrushStroke(event,point,imagePoint){
  const session=selectedBrushSession();
  if(exportOperationActive()){waitForExportOperation();return false}
  if(!session?.editor||selectedBrushBusy()){toast(session?.error||NO_SAM_MASK_MESSAGE,true);return false}
  if(!brushSessionStillCompatible(session,currentObject())){toast("Brush base changed. Reload or reset local edits before painting.",true);return false}
  const mode=store.activeTool==="manual-add"?"add":"remove";
  session.editor.begin(mode);
  session.editor.paint(imagePoint,imagePoint,Number($("#brush-size").value));
  brushPointer={pointerId:event.pointerId,last:imagePoint};
  stage.setPointerCapture(event.pointerId);
  updateBrushDirty();
  return true;
}

stage.addEventListener("pointerdown",event=>{
  if(!store.workspace||!sourceImage)return;
  const point=canvasEventPoint(canvas,event);
  if(store.activeTool==="pan"||event.button===1||event.altKey||event.shiftKey){stage.setPointerCapture(event.pointerId);panPointer=point;return}
  const imagePoint=canvasPointToImage(view,point.x,point.y);
  if(!imagePoint)return;
  if((store.activeTool==="manual-add"||store.activeTool==="manual-remove")&&currentObject()){
    beginBrushStroke(event,point,imagePoint);
    return;
  }
  if((store.activeTool==="positive"||store.activeTool==="negative")&&currentObject()){
    if(!ensureNoExportOperation())return;
    if(!ensureNoSelectedManualOperation())return;
    if(store.samStatus.state!=="ready"){toast("SAM is not ready yet.",true);return}
    if(structuralBusyByObject.get(store.selectedObjectId)){toast("Wait for the current object action to finish.",true);return}
    if(manualMaskActive(currentObject())){toast("Clear saved manual corrections before changing the SAM mask.",true);return}
    if(brushDirty()){toast("Save or reset your brush edits before changing the SAM base.",true);return}
    stage.setPointerCapture(event.pointerId);
    const label=store.activeTool==="positive"?1:0;
    const result=store.addPoint(store.selectedObjectId,{x:imagePoint.x,y:imagePoint.y,label});
    requestDraw();
    submitPrompt(store.selectedObjectId,result.revision,result.points,result.box);
  }
});
stage.addEventListener("pointermove",event=>{
  const point=canvasEventPoint(canvas,event);
  const imagePoint=canvasPointToImage(view,point.x,point.y);
  brushCursor=(store.activeTool==="manual-add"||store.activeTool==="manual-remove")&&imagePoint?point:null;
  if(panPointer){view.panBy(point.x-panPointer.x,point.y-panPointer.y);panPointer=point;requestDraw();return}
  if(brushPointer&&selectedBrushSession()?.editor){
    if(imagePoint){
      selectedBrushSession().editor.paint(brushPointer.last||imagePoint,imagePoint,Number($("#brush-size").value));
      brushPointer.last=imagePoint;
      updateBrushDirty();
    }else brushPointer.last=null;
    return;
  }
  requestDraw();
});
function finishBrushPointer(){
  if(brushPointer&&selectedBrushSession()?.editor&&selectedBrushSession().editor.commit())updateBrushDirty();
  brushPointer=null;panPointer=null;
}
stage.addEventListener("pointerup",finishBrushPointer);
stage.addEventListener("pointercancel",finishBrushPointer);
stage.addEventListener("pointerleave",()=>{brushCursor=null;requestDraw()});
stage.addEventListener("wheel",event=>{if(!store.workspace)return;event.preventDefault();const point=canvasEventPoint(canvas,event);view.zoomAt(event.deltaY<0?1.12:.89,point.x,point.y);updateZoom();requestDraw()},{passive:false});

$("#workspace-loader").addEventListener("submit",event=>{event.preventDefault();const id=$("#workspace-id").value.trim();if(id&&confirmDiscardBrushEdits("Discard unsaved brush edits and load another workspace?"))loadWorkspace(id)});
$("#upload-form").addEventListener("submit",async event=>{event.preventDefault();const file=$("#source-image").files[0];if(!file||!confirmDiscardBrushEdits("Discard unsaved brush edits and create another workspace?"))return;try{await createWorkspace(file)}catch(error){toast(error.message,true)}});
$("#retry-sam").addEventListener("click",prepareSam);
$("#new-object").addEventListener("click",()=>{if(!ensureNoExportOperation()||!ensureNoManualOperation())return;$("#object-create-form").hidden=false;$("#new-display-name").focus()});
$("#cancel-create").addEventListener("click",()=>{$("#object-create-form").hidden=true});
$("#new-display-name").addEventListener("input",()=>{$("#new-semantic-label").value=semanticLabelFromDisplayName($("#new-display-name").value)});
$("#object-create-form").addEventListener("submit",async event=>{event.preventDefault();if(!ensureNoExportOperation()||!ensureNoManualOperation())return;try{const display=$("#new-display-name").value.trim();const label=$("#new-semantic-label").value.trim()||semanticLabelFromDisplayName(display);const workspace=await api.createObject(store.workspace,label,display);const created=workspace.objects.find(o=>!store.workspace.objects.some(existing=>existing.object_id===o.object_id));$("#object-create-form").hidden=true;$("#new-display-name").value="";$("#new-semantic-label").value="";updateWorkspace(workspace,created?.object_id);toast("Object created")}catch(error){if(error.status===409)await reloadAfterConflict("Workspace changed. Reloaded; retry create.");else toast(error.message,true)}});
$("#object-list").addEventListener("click",event=>{const row=event.target.closest("[data-object-id]");if(row){if(row.dataset.objectId!==store.selectedObjectId&&!confirmDiscardBrushEdits("Discard unsaved brush edits and select another object?"))return;store.select(row.dataset.objectId);prepareBrushForSelection();renderPredictionStatus();requestDraw()}});
$("#rename-form").addEventListener("submit",async event=>{event.preventDefault();const object=currentObject();if(!object||!ensureNoExportOperation()||!ensureNoSelectedManualOperation())return;try{const workspace=await api.updateObject(store.workspace,object,$("#semantic-label").value.trim(),$("#display-name").value.trim());updateWorkspace(workspace,object.object_id);toast("Object renamed")}catch(error){if(error.status===409)await reloadAfterConflict("Workspace changed. Reloaded; retry rename.");else toast(error.message,true)}});
$("#delete-object").addEventListener("click",async()=>{const object=currentObject();if(!object||structuralBusyByObject.get(object.object_id)||!ensureNoExportOperation()||!ensureNoSelectedManualOperation())return;const extra=brushDirty()?" This will also discard unsaved brush edits.":"";if(!confirm(`Delete ${object.display_name}?${extra}`))return;const previousBrush=brushSession;structuralBusyByObject.set(object.object_id,"delete");disposeController(object.object_id);disposeBrushSession();renderAll();try{const workspace=await api.deleteObject(store.workspace,object);clearObjectMaskCache(store.workspace.workspace_id,object.object_id);const next=workspace.objects[0]?.object_id??null;updateWorkspace(workspace,next);toast("Object deleted")}catch(error){structuralBusyByObject.delete(object.object_id);store.resetPromptState(object.object_id);if(error.status===409)await reloadAfterConflict("Workspace changed. Reloaded; retry delete.");else{restoreBrushSession(previousBrush,store.workspace?.objects.find(item=>item.object_id===object.object_id));toast(error.message,true)}renderAll()}});
$("#candidate-list").addEventListener("click",async event=>{const button=event.target.closest("[data-candidate-index]");const object=currentObject();if(!button||!object||!ensureNoExportOperation()||!ensureNoSelectedManualOperation())return;const state=currentPromptState();const controller=selectedController();if(manualMaskActive(object)){toast("Clear saved manual corrections before changing the SAM mask.",true);return}if(brushDirty()){toast("Save or reset your brush edits before changing the SAM base.",true);return}if(controller?.running||controller?.pending||!canSelectCandidate(state)){toast("Wait for the current prediction before selecting a candidate.",true);return}const index=Number(button.dataset.candidateIndex);const previous=object.sam_draft.selected_candidate_index;const previousBrush=brushSession;object.sam_draft.selected_candidate_index=index;disposeBrushSession();renderDetails();requestDraw();try{const response=await api.selectCandidate(store.workspace,object,object.sam_draft.prompt_revision,index);updateWorkspace(mergeCandidateSelectionResponse(store.workspace,response),object.object_id);prepareBrushForSelection()}catch(error){object.sam_draft.selected_candidate_index=previous;renderDetails();requestDraw();if(!restoreBrushSession(previousBrush,object))prepareBrushForSelection();if(error.status===409)await reloadAfterConflict("Workspace changed. Reloaded; retry candidate selection.");else toast(error.message,true)}});
$("#undo-point").addEventListener("click",async()=>{const object=currentObject();if(!object||!ensureNoExportOperation()||!ensureNoSelectedManualOperation())return;if(manualMaskActive(object)){toast("Clear saved manual corrections before changing the SAM mask.",true);return}if(brushDirty()){toast("Save or reset your brush edits before changing the SAM base.",true);return}const result=store.undoPoint(object.object_id);if(!result)return;if(!result.points.length&&!result.box){await clearDraft(object.object_id);return}requestDraw();submitPrompt(object.object_id,result.revision,result.points,result.box)});
$("#reset-draft").addEventListener("click",()=>{if(currentObject())clearDraft(currentObject().object_id)});
$("#undo-brush").addEventListener("click",()=>{if(!ensureNoExportOperation()||!ensureNoSelectedManualOperation())return;if(selectedBrushSession()?.editor?.undo()){updateBrushDirty()}});
$("#redo-brush").addEventListener("click",()=>{if(!ensureNoExportOperation()||!ensureNoSelectedManualOperation())return;if(selectedBrushSession()?.editor?.redo()){updateBrushDirty()}});
$("#reset-brush").addEventListener("click",()=>{if(!ensureNoExportOperation()||!ensureNoSelectedManualOperation())return;selectedBrushSession()?.editor?.reset();updateBrushDirty()});
$("#save-manual").addEventListener("click",saveManualMask);
$("#clear-manual").addEventListener("click",clearManualMask);
$("#create-export").addEventListener("click",createExportSnapshot);
$("#refresh-exports").addEventListener("click",()=>refreshExportHistory(selectedWorkspaceId(),exportState.selectedExportId));
$("#export-history").addEventListener("click",event=>{
  const row=event.target.closest("[data-export-id]");
  if(!row)return;
  exportState.select(row.dataset.exportId);
  renderExports();
});
$("#export-details").addEventListener("toggle",event=>{
  if(event.target.id==="quality-details"&&event.target.open)loadSelectedQualityReport();
},true);
document.querySelectorAll("[data-tool]").forEach(button=>button.addEventListener("click",()=>{store.setTool(button.dataset.tool);document.querySelectorAll("[data-tool]").forEach(item=>item.classList.toggle("is-active",item===button));stage.dataset.tool=button.dataset.tool}));
$("#brush-size").addEventListener("input",()=>{$("#brush-size-output").textContent=`${$("#brush-size").value} px`;requestDraw()});
$("#overlay-opacity").addEventListener("input",()=>{$("#opacity-output").textContent=`${Math.round(Number($("#overlay-opacity").value)*100)}%`;requestDraw()});
$("#zoom-in").addEventListener("click",()=>{view.zoomAt(1.2,view.viewportWidth/2,view.viewportHeight/2);updateZoom();requestDraw()});
$("#zoom-out").addEventListener("click",()=>{view.zoomAt(.8,view.viewportWidth/2,view.viewportHeight/2);updateZoom();requestDraw()});
$("#fit-view").addEventListener("click",()=>{view.fit();updateZoom();requestDraw()});
window.addEventListener("keydown",event=>{
  const target=event.target;
  if(target&&["INPUT","TEXTAREA"].includes(target.tagName))return;
  if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==="z"&&!event.shiftKey){
    event.preventDefault();
    if(!ensureNoExportOperation()||!ensureNoSelectedManualOperation())return;
    if(selectedBrushSession()?.editor?.undo())updateBrushDirty();
  }else if((event.ctrlKey||event.metaKey)&&((event.key.toLowerCase()==="z"&&event.shiftKey)||event.key.toLowerCase()==="y")){
    event.preventDefault();
    if(!ensureNoExportOperation()||!ensureNoSelectedManualOperation())return;
    if(selectedBrushSession()?.editor?.redo())updateBrushDirty();
  }
});
window.addEventListener("beforeunload",event=>{if(brushDirty()){event.preventDefault();event.returnValue=""}});

store.subscribe(renderAll);
new ResizeObserver(resizeCanvas).observe(stage);
stage.dataset.tool="inspect";
$("#retry-sam").hidden=true;
const initialWorkspace=new URLSearchParams(location.search).get("workspace");
if(initialWorkspace){$("#workspace-id").value=initialWorkspace;loadWorkspace(initialWorkspace)}
