import {MaskEditor} from "./brush.js";
import {manualMaskActive,selectedCandidate} from "./segment-state.js";

export const NO_SAM_MASK_MESSAGE="Create a SAM mask before using manual corrections.";
export const MANUAL_BASE_MISMATCH_MESSAGE="Saved manual corrections are anchored to a different SAM candidate. Clear them or reload the matching SAM state before saving.";

export function manualBaseMatches(object){
  if(!manualMaskActive(object))return true;
  const draft=object.sam_draft||{},manual=object.manual_mask||{};
  return manual.base_prompt_revision===draft.prompt_revision&&manual.base_candidate_index===draft.selected_candidate_index;
}

export function brushBaseDescriptor(workspace,object){
  if(!workspace||!object)return {available:false,error:"Select an object before using manual corrections."};
  const draft=object.sam_draft||{},manual=object.manual_mask||{},candidate=selectedCandidate(object);
  if(manualMaskActive(object)){
    if(!manualBaseMatches(object)){
      return {available:false,error:MANUAL_BASE_MISMATCH_MESSAGE,manualRevision:manual.manual_revision||0};
    }
    return {
      available:true,
      maskType:"manual-composite",
      sourceMaskUrl:manual.composite_mask_url,
      basePromptRevision:manual.base_prompt_revision,
      baseCandidateIndex:manual.base_candidate_index,
      manualRevision:manual.manual_revision||0
    };
  }
  if(!candidate||draft.selected_candidate_index===null||draft.selected_candidate_index===undefined){
    return {available:false,error:NO_SAM_MASK_MESSAGE,manualRevision:0};
  }
  return {
    available:true,
    maskType:"sam-candidate",
    sourceMaskUrl:candidate.mask_url,
    basePromptRevision:draft.prompt_revision||0,
    baseCandidateIndex:draft.selected_candidate_index,
    manualRevision:0
  };
}

export function brushIdentity({workspaceId,objectId,maskType,basePromptRevision,baseCandidateIndex,manualRevision,sourceMaskUrl}){
  return `${workspaceId}:${objectId}:${maskType}:${basePromptRevision}:${baseCandidateIndex}:${manualRevision}:${sourceMaskUrl}`;
}

export function createBrushSession({workspace,object,sourceWidth,sourceHeight,mask,generation}){
  const descriptor=brushBaseDescriptor(workspace,object);
  if(!descriptor.available)return {...descriptor,workspaceId:workspace?.workspace_id??null,objectId:object?.object_id??null,sourceWidth,sourceHeight,generation,editor:null,editRevision:0,loading:false,saving:false,clearing:false};
  const session={
    workspaceId:workspace.workspace_id,
    objectId:object.object_id,
    sourceWidth,
    sourceHeight,
    basePromptRevision:descriptor.basePromptRevision,
    baseCandidateIndex:descriptor.baseCandidateIndex,
    manualRevision:descriptor.manualRevision,
    maskType:descriptor.maskType,
    sourceMaskUrl:descriptor.sourceMaskUrl,
    identity:brushIdentity({workspaceId:workspace.workspace_id,objectId:object.object_id,...descriptor}),
    editor:new MaskEditor(mask,sourceWidth,sourceHeight),
    editRevision:0,
    loading:false,
    saving:false,
    clearing:false,
    generation,
    error:""
  };
  return session;
}

export function brushSessionMatches(session,workspace,object,generation=session?.generation){
  if(!session||!workspace||!object)return false;
  const descriptor=brushBaseDescriptor(workspace,object);
  if(!descriptor.available)return false;
  return session.generation===generation&&session.workspaceId===workspace.workspace_id&&session.objectId===object.object_id&&session.basePromptRevision===descriptor.basePromptRevision&&session.baseCandidateIndex===descriptor.baseCandidateIndex&&session.manualRevision===descriptor.manualRevision&&session.sourceMaskUrl===descriptor.sourceMaskUrl;
}

export function rebaseBrushSession(session,mask=session?.editor?.mask){
  if(!session?.editor||!mask)return null;
  session.editor=new MaskEditor(mask,session.sourceWidth,session.sourceHeight);
  session.editRevision+=1;
  return session;
}

export function markBrushEdited(session){
  if(session)session.editRevision+=1;
  return session;
}

export function binaryMaskRgba(mask,width,height){
  const data=new Uint8ClampedArray(width*height*4);
  for(let i=0;i<width*height;i++){
    const value=mask[i]?255:0;
    data[i*4]=value;
    data[i*4+1]=value;
    data[i*4+2]=value;
    data[i*4+3]=255;
  }
  return data;
}

export function binaryMaskToPngBlob(mask,width,height){
  const canvas=document.createElement("canvas");
  canvas.width=width;
  canvas.height=height;
  const context=canvas.getContext("2d");
  context.putImageData(new ImageData(binaryMaskRgba(mask,width,height),width,height),0,0);
  return new Promise((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(new Error("Could not encode mask PNG")),"image/png"));
}

export function brushSaveState(session){
  if(!session?.editor)return session?.error||NO_SAM_MASK_MESSAGE;
  if(session.saving)return "Saving...";
  if(session.clearing)return "Clearing...";
  if(session.error)return session.error;
  return session.editor.dirty?"Unsaved brush edits":"Saved";
}
