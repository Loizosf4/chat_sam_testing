import {ViewportTransform} from "/static/transform.js";
import {binaryFromImageData,colorForIndex,rgbaForMask} from "/static/mask-utils.js";
import {SegmentationWorkspaceClient} from "/static/segment-api.js";
import {canvasEventPoint,canvasPointToImage,fitViewport} from "/static/segment-coordinates.js";
import {candidateListHtml,objectListHtml} from "/static/segment-components.js";
import {
  PredictionController,
  SegmentStore,
  canSelectCandidate,
  mergeCandidateSelectionResponse,
  mergeClearDraftResponse,
  mergePredictionResponse,
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
const controllers=new Map();
const candidateImageCache=new Map();
let sourceImage=null,renderQueued=false,pointer=null,panPointer=null;

function absoluteUrl(path){return new URL(path,location.origin).href}
function toast(message,error=false){const el=document.createElement("div");el.className=`toast${error?" error":""}`;el.textContent=message;$("#status-region").append(el);setTimeout(()=>el.remove(),4200)}
function setCanvasMessage(message){$("#canvas-message").textContent=message;$("#canvas-message").hidden=!message}
function currentObject(){return store.selected}
function currentPromptState(){return store.getPromptState(store.selectedObjectId)}

function imageElement(url){return new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(new Error(`Could not load ${url}`));img.src=absoluteUrl(url)})}
async function maskBinary(candidate,draft){if(!candidate)return null;const key=`${candidate.mask_url}?prompt_revision=${draft.prompt_revision}&candidate=${candidate.candidate_index}`;if(candidateImageCache.has(key))return candidateImageCache.get(key);const img=await imageElement(key);const c=document.createElement("canvas");c.width=view.imageWidth;c.height=view.imageHeight;const cctx=c.getContext("2d",{willReadFrequently:true});cctx.drawImage(img,0,0,c.width,c.height);const binary=binaryFromImageData(cctx.getImageData(0,0,c.width,c.height));candidateImageCache.set(key,binary);return binary}

async function loadSourceImage(workspace){sourceImage=await imageElement(workspace.source_image.url);view.setImageSize(workspace.image_dimensions.width,workspace.image_dimensions.height).fit();resizeCanvas()}
async function prepareSam(){if(!store.workspace)return;store.setSamStatus("preparing","Preparing...");try{const result=await api.prepareSam(store.workspace.workspace_id);store.setSamStatus("ready",`Ready (${result.model_type} ${result.device})`)}catch(error){store.setSamStatus("failed",error.message);toast(error.message,true)}}
async function loadWorkspace(workspaceId){try{store.loadingState="Loading";store.setError("");setCanvasMessage("Loading workspace...");const workspace=await api.getWorkspace(workspaceId);await loadSourceImage(workspace);store.load(workspace);history.replaceState(null,"",`/segment?workspace=${encodeURIComponent(workspace.workspace_id)}`);$("#workspace-id").value=workspace.workspace_id;$("#empty-state").hidden=true;$("#workspace-app").hidden=false;setCanvasMessage("");await prepareSam()}catch(error){setCanvasMessage("");toast(error.message,true);store.setError(error.message)}}
async function createWorkspace(file){const workspace=await api.createWorkspace(file);await loadSourceImage(workspace);store.load(workspace);history.replaceState(null,"",`/segment?workspace=${encodeURIComponent(workspace.workspace_id)}`);$("#workspace-id").value=workspace.workspace_id;$("#empty-state").hidden=true;$("#workspace-app").hidden=false;await prepareSam()}

function updateWorkspace(workspace,preserveId=store.selectedObjectId){store.load(workspace,preserveId);requestDraw()}
function selectedObjectColor(){const index=Math.max(0,(store.workspace?.objects||[]).findIndex(o=>o.object_id===store.selectedObjectId));return colorForIndex(index)}

function renderObjects(){if(!store.workspace)return;$("#object-list").innerHTML=objectListHtml(store.workspace.objects,store.selectedObjectId,store.localPromptStateByObject)}
function renderDetails(){
  const object=currentObject();
  $("#rename-form").hidden=!object;
  $("#delete-object").disabled=!object;
  const samReady=store.samStatus.state==="ready";
  document.querySelectorAll("[data-tool='positive'],[data-tool='negative']").forEach(button=>button.disabled=!samReady);
  $("#undo-point").disabled=!object||!currentPromptState()?.localPoints.length;
  $("#reset-draft").disabled=!object;
  if(!object){$("#candidate-list").innerHTML="<p class='muted-pad'>Select an object</p>";return}
  $("#display-name").value=object.display_name;
  $("#semantic-label").value=object.semantic_label;
  const prompt=currentPromptState();
  $("#point-count").textContent=prompt?.localPoints.length??0;
  $("#prompt-revision").textContent=object.sam_draft?.prompt_revision??0;
  $("#object-version").textContent=object.object_version;
  $("#candidate-list").innerHTML=candidateListHtml(object,object.sam_draft?.selected_candidate_index);
}
function renderSamStatus(){const el=$("#sam-status");el.dataset.state=store.samStatus.state;el.querySelector("output").textContent=store.samStatus.message;$("#retry-sam").hidden=!store.workspace}
function renderAll(){renderObjects();renderDetails();renderSamStatus();requestDraw()}

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
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,width,height);
  if(!sourceImage||!store.workspace)return;
  const origin=view.origin,scale=view.scale;
  ctx.drawImage(sourceImage,origin.x,origin.y,view.imageWidth*scale,view.imageHeight*scale);
  const object=currentObject();
  if(object){
    const draft=object.sam_draft||{};
    const candidate=selectedCandidate(object);
    const mask=await maskBinary(candidate,draft).catch(error=>{toast(`Candidate artifact failed to load: ${error.message}`,true);return null});
    if(mask){
      const overlay=document.createElement("canvas");
      overlay.width=view.imageWidth;
      overlay.height=view.imageHeight;
      overlay.getContext("2d").putImageData(rgbaForMask(mask,view.imageWidth,view.imageHeight,selectedObjectColor(),Number($("#overlay-opacity").value),true),0,0);
      ctx.imageSmoothingEnabled=false;
      ctx.drawImage(overlay,origin.x,origin.y,view.imageWidth*scale,view.imageHeight*scale);
    }
    for(const point of currentPromptState()?.localPoints||draft.points||[])drawPoint(point);
  }
}
function requestDraw(){if(!renderQueued){renderQueued=true;requestAnimationFrame(draw)}}
function resizeCanvas(){const rect=stage.getBoundingClientRect();fitViewport(view,rect.width,rect.height,window.devicePixelRatio||1);requestDraw();updateZoom()}
function updateZoom(){$("#zoom-level").textContent=`${Math.round(view.zoom*100)}%`}

function getController(objectId){
  if(!controllers.has(objectId)){
    controllers.set(objectId,new PredictionController({
      objectId,
      isActive:id=>store.workspace?.objects.some(o=>o.object_id===id),
      submit:snapshot=>api.predict(store.workspace.workspace_id,objectId,{
        prompt_revision:snapshot.revision,
        points:snapshot.points,
        box:snapshot.box,
        base_prompt_revision:snapshot.basePromptRevision,
        base_candidate_index:snapshot.baseCandidateIndex,
        multimask_output:null
      }),
      apply:(response,snapshot)=>{
        if(!store.workspace||!store.workspace.objects.some(o=>o.object_id===response.object_id))return;
        updateWorkspace(mergePredictionResponse(store.workspace,response),store.selectedObjectId);
        const state=store.getPromptState(response.object_id);
        state.latestAppliedRevision=response.prompt_revision;
        state.persistedRevision=response.prompt_revision;
        state.localPoints=[...snapshot.points];
        state.running=false;
        state.pending=false;
        $("#mask-updating").hidden=true;
      },
      onError:(error,snapshot)=>{
        $("#mask-updating").hidden=true;
        const state=store.getPromptState(snapshot.objectId);
        if(state){state.running=false;state.pending=false}
        if(error.status===409&&state?.nextRevision>snapshot.revision+1)return;
        if(error.status===409)reloadAfterConflict("Prediction was stale. Reloaded workspace state.");
        else toast(error.message,true);
      }
    }));
  }
  return controllers.get(objectId);
}

function submitPrompt(objectId,revision,points,box){
  const object=store.workspace.objects.find(o=>o.object_id===objectId);
  const draft=object?.sam_draft||{};
  const hasBase=Number.isInteger(draft.prompt_revision)&&draft.prompt_revision>0&&draft.selected_candidate_index!==null&&draft.selected_candidate_index!==undefined;
  const controller=getController(objectId);
  const promptState=store.getPromptState(objectId);
  $("#mask-updating").hidden=false;
  controller.enqueue({
    objectId,
    revision,
    points:points.map(p=>({...p})),
    box,
    basePromptRevision:hasBase?draft.prompt_revision:null,
    baseCandidateIndex:hasBase?draft.selected_candidate_index:null
  });
  promptState.running=controller.running;
  promptState.pending=Boolean(controller.pending);
}

async function clearDraft(objectId){
  const object=store.workspace.objects.find(o=>o.object_id===objectId);
  if(!object)return;
  controllers.get(objectId)?.invalidate();
  try{
    const response=await api.clearDraft(store.workspace,object);
    updateWorkspace(mergeClearDraftResponse(store.workspace,response),objectId);
    store.resetPromptState(objectId);
    candidateImageCache.clear();
    toast("SAM draft reset");
  }catch(error){if(error.status===409)await reloadAfterConflict("Workspace changed. Reloaded; retry reset.");else toast(error.message,true)}
}

async function reloadAfterConflict(message){const id=store.workspace?.workspace_id;if(id){const workspace=await api.getWorkspace(id);updateWorkspace(workspace,store.selectedObjectId);toast(message,true)}}

stage.addEventListener("pointerdown",event=>{
  if(!store.workspace||!sourceImage)return;
  const point=canvasEventPoint(canvas,event);
  stage.setPointerCapture(event.pointerId);
  if(store.activeTool==="pan"||event.button===1||event.altKey||event.shiftKey){panPointer=point;return}
  const imagePoint=canvasPointToImage(view,point.x,point.y);
  if(!imagePoint)return;
  if((store.activeTool==="positive"||store.activeTool==="negative")&&currentObject()){
    if(store.samStatus.state!=="ready"){toast("SAM is not ready yet.",true);return}
    const label=store.activeTool==="positive"?1:0;
    const result=store.addPoint(store.selectedObjectId,{x:imagePoint.x,y:imagePoint.y,label});
    requestDraw();
    submitPrompt(store.selectedObjectId,result.revision,result.points,result.box);
  }
});
stage.addEventListener("pointermove",event=>{if(!panPointer)return;const point=canvasEventPoint(canvas,event);view.panBy(point.x-panPointer.x,point.y-panPointer.y);panPointer=point;requestDraw()});
stage.addEventListener("pointerup",()=>{panPointer=null});
stage.addEventListener("wheel",event=>{if(!store.workspace)return;event.preventDefault();const point=canvasEventPoint(canvas,event);view.zoomAt(event.deltaY<0?1.12:.89,point.x,point.y);updateZoom();requestDraw()},{passive:false});

$("#workspace-loader").addEventListener("submit",event=>{event.preventDefault();const id=$("#workspace-id").value.trim();if(id)loadWorkspace(id)});
$("#upload-form").addEventListener("submit",async event=>{event.preventDefault();const file=$("#source-image").files[0];if(!file)return;try{await createWorkspace(file)}catch(error){toast(error.message,true)}});
$("#retry-sam").addEventListener("click",prepareSam);
$("#new-object").addEventListener("click",()=>{$("#object-create-form").hidden=false;$("#new-display-name").focus()});
$("#cancel-create").addEventListener("click",()=>{$("#object-create-form").hidden=true});
$("#new-display-name").addEventListener("input",()=>{$("#new-semantic-label").value=semanticLabelFromDisplayName($("#new-display-name").value)});
$("#object-create-form").addEventListener("submit",async event=>{event.preventDefault();try{const display=$("#new-display-name").value.trim();const label=$("#new-semantic-label").value.trim()||semanticLabelFromDisplayName(display);const workspace=await api.createObject(store.workspace,label,display);const created=workspace.objects.find(o=>!store.workspace.objects.some(existing=>existing.object_id===o.object_id));$("#object-create-form").hidden=true;$("#new-display-name").value="";$("#new-semantic-label").value="";updateWorkspace(workspace,created?.object_id);toast("Object created")}catch(error){if(error.status===409)await reloadAfterConflict("Workspace changed. Reloaded; retry create.");else toast(error.message,true)}});
$("#object-list").addEventListener("click",event=>{const row=event.target.closest("[data-object-id]");if(row){store.select(row.dataset.objectId);requestDraw()}});
$("#rename-form").addEventListener("submit",async event=>{event.preventDefault();const object=currentObject();if(!object)return;try{const workspace=await api.updateObject(store.workspace,object,$("#semantic-label").value.trim(),$("#display-name").value.trim());updateWorkspace(workspace,object.object_id);toast("Object renamed")}catch(error){if(error.status===409)await reloadAfterConflict("Workspace changed. Reloaded; retry rename.");else toast(error.message,true)}});
$("#delete-object").addEventListener("click",async()=>{const object=currentObject();if(!object||!confirm(`Delete ${object.display_name}?`))return;controllers.get(object.object_id)?.invalidate();try{const workspace=await api.deleteObject(store.workspace,object);const next=workspace.objects[0]?.object_id??null;updateWorkspace(workspace,next);toast("Object deleted")}catch(error){if(error.status===409)await reloadAfterConflict("Workspace changed. Reloaded; retry delete.");else toast(error.message,true)}});
$("#candidate-list").addEventListener("click",async event=>{const button=event.target.closest("[data-candidate-index]");const object=currentObject();if(!button||!object)return;const state=currentPromptState();const controller=controllers.get(object.object_id);if(controller?.running||controller?.pending||!canSelectCandidate(state)){toast("Wait for the current prediction before selecting a candidate.",true);return}const index=Number(button.dataset.candidateIndex);const previous=object.sam_draft.selected_candidate_index;object.sam_draft.selected_candidate_index=index;renderDetails();requestDraw();try{const response=await api.selectCandidate(store.workspace,object,object.sam_draft.prompt_revision,index);updateWorkspace(mergeCandidateSelectionResponse(store.workspace,response),object.object_id)}catch(error){object.sam_draft.selected_candidate_index=previous;renderDetails();requestDraw();if(error.status===409)await reloadAfterConflict("Workspace changed. Reloaded; retry candidate selection.");else toast(error.message,true)}});
$("#undo-point").addEventListener("click",async()=>{const object=currentObject();if(!object)return;const result=store.undoPoint(object.object_id);if(!result)return;if(!result.points.length&&!result.box){await clearDraft(object.object_id);return}requestDraw();submitPrompt(object.object_id,result.revision,result.points,result.box)});
$("#reset-draft").addEventListener("click",()=>{if(currentObject())clearDraft(currentObject().object_id)});
document.querySelectorAll("[data-tool]").forEach(button=>button.addEventListener("click",()=>{store.setTool(button.dataset.tool);document.querySelectorAll("[data-tool]").forEach(item=>item.classList.toggle("is-active",item===button));stage.dataset.tool=button.dataset.tool}));
$("#overlay-opacity").addEventListener("input",()=>{$("#opacity-output").textContent=`${Math.round(Number($("#overlay-opacity").value)*100)}%`;requestDraw()});
$("#zoom-in").addEventListener("click",()=>{view.zoomAt(1.2,view.viewportWidth/2,view.viewportHeight/2);updateZoom();requestDraw()});
$("#zoom-out").addEventListener("click",()=>{view.zoomAt(.8,view.viewportWidth/2,view.viewportHeight/2);updateZoom();requestDraw()});
$("#fit-view").addEventListener("click",()=>{view.fit();updateZoom();requestDraw()});

store.subscribe(renderAll);
new ResizeObserver(resizeCanvas).observe(stage);
stage.dataset.tool="inspect";
$("#retry-sam").hidden=true;
const initialWorkspace=new URLSearchParams(location.search).get("workspace");
if(initialWorkspace){$("#workspace-id").value=initialWorkspace;loadWorkspace(initialWorkspace)}
