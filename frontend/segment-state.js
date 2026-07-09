export function semanticLabelFromDisplayName(value){
  return String(value||"").trim().toLowerCase().replace(/[^a-z0-9]+/g,"_").replace(/^_+|_+$/g,"")||"object";
}

export function emptyDraft(){
  return {prompt_revision:0,points:[],box:null,candidates:[],selected_candidate_index:null,prepared_image_key:null,updated_at:null};
}

export function emptyManualMask(){
  return {manual_revision:0,base_prompt_revision:null,base_candidate_index:null,add_mask_url:null,remove_mask_url:null,composite_mask_url:null,area_pixels:0,bbox_xyxy:[0,0,0,0],updated_at:null};
}

export function normalizeObject(object){
  return {...object,sam_draft:{...emptyDraft(),...(object.sam_draft||{})},manual_mask:{...emptyManualMask(),...(object.manual_mask||{})}};
}

export function normalizeWorkspace(workspace){
  return {...workspace,objects:(workspace.objects||[]).map(normalizeObject)};
}

export function createPromptState(object){
  const draft=normalizeObject(object).sam_draft;
  return {
    objectId:object.object_id,
    persistedRevision:draft.prompt_revision||0,
    localPoints:[...(draft.points||[])],
    box:draft.box??null,
    nextRevision:(draft.prompt_revision||0)+1,
    latestSubmittedRevision:0,
    latestAppliedRevision:draft.prompt_revision||0,
    running:false,
    pending:false,
    invalidated:false
  };
}

export function clonePoint(point){return{x:Number(point.x),y:Number(point.y),label:Number(point.label)}}

export class SegmentStore {
  constructor(){
    this.workspace=null;
    this.selectedObjectId=null;
    this.activeTool="inspect";
    this.overlayOpacity=.55;
    this.samStatus={state:"idle",message:"Not prepared"};
    this.loadingState="";
    this.errorState="";
    this.localPromptStateByObject=new Map();
    this.listeners=new Set();
  }
  subscribe(listener){this.listeners.add(listener);return()=>this.listeners.delete(listener)}
  emit(){for(const listener of this.listeners)listener(this)}
  load(workspace,preserveId=this.selectedObjectId){
    this.workspace=normalizeWorkspace(workspace);
    this.selectedObjectId=this.workspace.objects.some(o=>o.object_id===preserveId)?preserveId:this.workspace.objects[0]?.object_id??null;
    this.syncPromptStates();
    this.emit();
  }
  syncPromptStates(){
    const validIds=new Set((this.workspace?.objects||[]).map(o=>o.object_id));
    for(const key of [...this.localPromptStateByObject.keys()])if(!validIds.has(key))this.localPromptStateByObject.delete(key);
    for(const object of this.workspace?.objects||[]){
      if(!this.localPromptStateByObject.has(object.object_id))this.localPromptStateByObject.set(object.object_id,createPromptState(object));
      else{
        const state=this.localPromptStateByObject.get(object.object_id);
        const revision=object.sam_draft?.prompt_revision||0;
        if(!state.running&&!state.pending&&revision>=state.latestAppliedRevision){
          state.persistedRevision=revision;
          state.latestAppliedRevision=revision;
          state.localPoints=[...(object.sam_draft?.points||[])];
          state.box=object.sam_draft?.box??null;
          state.nextRevision=Math.max(state.nextRevision,revision+1);
        }
      }
    }
  }
  get selected(){return this.workspace?.objects.find(o=>o.object_id===this.selectedObjectId)??null}
  getPromptState(objectId=this.selectedObjectId){
    if(!objectId)return null;
    const object=this.workspace?.objects.find(o=>o.object_id===objectId);
    if(!object)return null;
    if(!this.localPromptStateByObject.has(objectId))this.localPromptStateByObject.set(objectId,createPromptState(object));
    return this.localPromptStateByObject.get(objectId);
  }
  select(objectId){if(this.workspace?.objects.some(o=>o.object_id===objectId)){this.selectedObjectId=objectId;this.emit()}}
  setTool(tool){this.activeTool=tool;this.emit()}
  setSamStatus(state,message){this.samStatus={state,message};this.emit()}
  setError(message){this.errorState=message||"";this.emit()}
  addPoint(objectId,point){
    const state=this.getPromptState(objectId);
    state.localPoints=[...state.localPoints,clonePoint(point)];
    const revision=this.nextPromptRevision(objectId);
    this.emit();
    return {revision,points:[...state.localPoints],box:state.box};
  }
  undoPoint(objectId){
    const state=this.getPromptState(objectId);
    if(!state||!state.localPoints.length)return null;
    state.localPoints=state.localPoints.slice(0,-1);
    const revision=this.nextPromptRevision(objectId);
    this.emit();
    return {revision,points:[...state.localPoints],box:state.box};
  }
  nextPromptRevision(objectId){
    const state=this.getPromptState(objectId);
    const revision=state.nextRevision;
    state.nextRevision+=1;
    return revision;
  }
  resetPromptState(objectId){
    const object=this.workspace?.objects.find(o=>o.object_id===objectId);
    if(object)this.localPromptStateByObject.set(objectId,createPromptState(object));
    this.emit();
  }
  replaceObject(object){this.load({...this.workspace,objects:this.workspace.objects.map(o=>o.object_id===object.object_id?normalizeObject(object):o)},object.object_id)}
}

export function mergePredictionResponse(workspace,response){
  const normalized=normalizeWorkspace(workspace);
  return {
    ...normalized,
    workspace_revision:response.workspace_revision,
    objects:normalized.objects.map(object=>{
      if(object.object_id!==response.object_id)return object;
      return normalizeObject({
        ...object,
        object_version:response.object_version,
        sam_draft:{
          ...object.sam_draft,
          prompt_revision:response.prompt_revision,
          points:response.points||[],
          box:response.box??null,
          candidates:response.candidates||[],
          selected_candidate_index:response.selected_candidate_index
        }
      });
    })
  };
}

export function mergeCandidateSelectionResponse(workspace,response){
  const normalized=normalizeWorkspace(workspace);
  return {
    ...normalized,
    workspace_revision:response.workspace_revision,
    objects:normalized.objects.map(object=>object.object_id===response.object_id?normalizeObject({
      ...object,
      object_version:response.object_version,
      sam_draft:{...object.sam_draft,selected_candidate_index:response.selected_candidate_index,candidates:response.candidates||object.sam_draft.candidates,prompt_revision:response.prompt_revision}
    }):object)
  };
}

export function mergeClearDraftResponse(workspace,response){
  const normalized=normalizeWorkspace(workspace);
  return {
    ...normalized,
    workspace_revision:response.workspace_revision,
    objects:normalized.objects.map(object=>object.object_id===response.object_id?normalizeObject({...object,object_version:response.object_version,sam_draft:response.sam_draft||emptyDraft(),manual_mask:emptyManualMask()}):object)
  };
}

export function manualMaskActive(object){
  return Number(normalizeObject(object).manual_mask.manual_revision)>0;
}

export function mergeManualMaskResponse(workspace,response){
  const normalized=normalizeWorkspace(workspace);
  return {
    ...normalized,
    workspace_revision:response.workspace_revision,
    objects:normalized.objects.map(object=>object.object_id===response.object_id?normalizeObject({
      ...object,
      object_version:response.object_version,
      manual_mask:response.manual_mask||emptyManualMask()
    }):object)
  };
}

export function selectedCandidate(object){
  const draft=normalizeObject(object).sam_draft;
  return draft.candidates.find(c=>c.candidate_index===draft.selected_candidate_index)??draft.candidates[0]??null;
}

export function canSelectCandidate(promptState){return Boolean(promptState)&&!promptState.running&&!promptState.pending}

export function canUseSamPrompting({object,promptState,brushDirty=false,structuralBusy=false,samReady=true}={}){
  return Boolean(object)&&samReady&&!structuralBusy&&!brushDirty&&!manualMaskActive(object)&&!promptState?.running&&!promptState?.pending;
}

export function canSelectCandidateForObject({object,promptState,brushDirty=false,structuralBusy=false}={}){
  return Boolean(object)&&!structuralBusy&&!brushDirty&&!manualMaskActive(object)&&canSelectCandidate(promptState);
}

export function predictionStatusVisible({controller=null,promptState=null}={}){
  return Boolean(controller?.running||controller?.pending||promptState?.running||promptState?.pending);
}

export function controllerKey(workspaceId,objectId){return `${workspaceId}:${objectId}`}

export function candidateCacheKey({workspaceId,objectId,promptRevision,candidateIndex,maskUrl}){
  return `${workspaceId}:${objectId}:${promptRevision}:${candidateIndex}:${maskUrl}`;
}

export function effectiveMaskCacheKey({workspaceId,objectId,maskType,promptRevision,candidateIndex,manualRevision,maskUrl}){
  return `${workspaceId}:${objectId}:${maskType}:${promptRevision??0}:${candidateIndex??"none"}:${manualRevision??0}:${maskUrl}`;
}

export class CandidateMaskCache {
  constructor(){this.items=new Map()}
  get(key){return this.items.get(key)}
  set(key,value){this.items.set(key,value);return value}
  has(key){return this.items.has(key)}
  clear(){this.items.clear()}
  clearObject(workspaceId,objectId){
    const prefix=`${workspaceId}:${objectId}:`;
    for(const key of [...this.items.keys()])if(key.startsWith(prefix))this.items.delete(key);
  }
}

export class WorkspaceLoadGate {
  constructor(){this.generation=0}
  begin(){this.generation+=1;return this.generation}
  isCurrent(generation){return generation===this.generation}
}

export class RenderGate {
  constructor(){this.generation=0}
  request(){this.generation+=1;return this.generation}
  snapshot(state){
    return {
      generation:this.generation,
      workspaceId:state.workspaceId??null,
      selectedObjectId:state.selectedObjectId??null,
      promptRevision:state.promptRevision??0,
      selectedCandidateIndex:state.selectedCandidateIndex??null,
      maskUrl:state.maskUrl??null,
      manualRevision:state.manualRevision??0,
      brushGeneration:state.brushGeneration??0,
      brushEditRevision:state.brushEditRevision??0,
      effectiveMaskIdentity:state.effectiveMaskIdentity??null,
      viewport:{...(state.viewport||{})},
      sourceImageId:state.sourceImageId??null
    };
  }
  isCurrent(snapshot,state){
    return Boolean(snapshot)&&snapshot.generation===this.generation&&
      snapshot.workspaceId===(state.workspaceId??null)&&
      snapshot.selectedObjectId===(state.selectedObjectId??null)&&
      snapshot.promptRevision===(state.promptRevision??0)&&
      snapshot.selectedCandidateIndex===(state.selectedCandidateIndex??null)&&
      snapshot.maskUrl===(state.maskUrl??null)&&
      snapshot.manualRevision===(state.manualRevision??0)&&
      snapshot.brushGeneration===(state.brushGeneration??0)&&
      snapshot.brushEditRevision===(state.brushEditRevision??0)&&
      snapshot.effectiveMaskIdentity===(state.effectiveMaskIdentity??null)&&
      snapshot.sourceImageId===(state.sourceImageId??null)&&
      JSON.stringify(snapshot.viewport)===JSON.stringify(state.viewport||{});
  }
}

export class ControllerRegistry {
  constructor(){this.controllers=new Map()}
  key(workspaceId,objectId){return controllerKey(workspaceId,objectId)}
  get(workspaceId,objectId){return this.controllers.get(this.key(workspaceId,objectId))}
  set(workspaceId,objectId,controller){this.controllers.set(this.key(workspaceId,objectId),controller);return controller}
  dispose(workspaceId,objectId){
    const key=this.key(workspaceId,objectId);
    const controller=this.controllers.get(key);
    controller?.invalidate();
    this.controllers.delete(key);
  }
  disposeWorkspace(workspaceId){
    for(const [key,controller] of [...this.controllers.entries()]){
      if(key.startsWith(`${workspaceId}:`)){controller.invalidate();this.controllers.delete(key)}
    }
  }
  clear(){for(const controller of this.controllers.values())controller.invalidate();this.controllers.clear()}
  has(workspaceId,objectId){return this.controllers.has(this.key(workspaceId,objectId))}
}

export class PredictionController {
  constructor({workspaceId=null,objectId,submit,apply,onError=()=>{},isActive=()=>true,onStateChange=()=>{}}){
    this.workspaceId=workspaceId;
    this.objectId=objectId;
    this.submit=submit;
    this.apply=apply;
    this.onError=onError;
    this.isActive=isActive;
    this.onStateChange=onStateChange;
    this.pending=null;
    this.running=false;
    this.invalidated=false;
    this.latestAppliedRevision=0;
    this.submitted=[];
  }
  _notify(){this.onStateChange({workspaceId:this.workspaceId,objectId:this.objectId,running:this.running,pending:Boolean(this.pending),invalidated:this.invalidated})}
  enqueue(snapshot){
    if(this.invalidated)return;
    this.pending={...snapshot,workspaceId:this.workspaceId,objectId:this.objectId};
    this._notify();
    if(!this.running)this._drain();
  }
  invalidate(){this.invalidated=true;this.pending=null;this.running=false;this._notify()}
  async _drain(){
    if(this.running||!this.pending||this.invalidated)return;
    const snapshot=this.pending;
    this.pending=null;
    this.running=true;
    this._notify();
    this.submitted.push(snapshot);
    try{
      const response=await this.submit(snapshot);
      if(this.invalidated)return;
      const hasNewerPending=Boolean(this.pending&&this.pending.revision>snapshot.revision);
      if(!hasNewerPending&&this.isActive(this.objectId)&&response.object_id===this.objectId&&response.prompt_revision===snapshot.revision&&response.prompt_revision>=this.latestAppliedRevision){
        this.latestAppliedRevision=response.prompt_revision;
        this.apply(response,snapshot);
      }
    }catch(error){
      if(this.invalidated)return;
      if(error.status===409&&this.pending&&this.pending.revision>snapshot.revision)return;
      this.onError(error,snapshot);
    }finally{
      this.running=false;
      this._notify();
      if(!this.invalidated&&this.pending)this._drain();
    }
  }
}

export async function persistCandidateSelectionOptimistic({workspace,objectId,candidateIndex,persist}){
  const object=normalizeWorkspace(workspace).objects.find(o=>o.object_id===objectId);
  if(!object)throw new Error("Object not found");
  const previous=object.sam_draft.selected_candidate_index;
  object.sam_draft.selected_candidate_index=candidateIndex;
  try{return await persist()}
  catch(error){object.sam_draft.selected_candidate_index=previous;throw error}
}
