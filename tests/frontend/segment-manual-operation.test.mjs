import test from "node:test";
import assert from "node:assert/strict";
import {
  ManualOperationRegistry,
  canStartManualOperation,
  createBrushSession,
  createManualOperationContext,
  manualConflictCompatibility,
  manualOperationMatchesCurrentState
} from "../../frontend/segment-brush.js";

function workspace(overrides={}){
  return {
    workspace_id:"workspace-a",
    workspace_revision:7,
    image_dimensions:{width:2,height:2},
    objects:[object()],
    ...overrides
  };
}

function object(overrides={}){
  return {
    object_id:"object-a",
    object_version:4,
    sam_draft:{prompt_revision:3,selected_candidate_index:0,candidates:[{candidate_index:0,mask_url:"/sam/0"}]},
    manual_mask:{manual_revision:1,base_prompt_revision:3,base_candidate_index:0,composite_mask_url:"/manual/1"},
    ...overrides
  };
}

function session(workspaceValue=workspace(),objectValue=object()){
  const mask=new Uint8Array([1,0,0,1]);
  const created=createBrushSession({workspace:workspaceValue,object:objectValue,sourceWidth:2,sourceHeight:2,mask,generation:11});
  created.editor.mask[1]=1;
  created.editRevision=5;
  return created;
}

test("manual operation context captures immutable workspace, object, session, and edited pixels",()=>{
  const w=workspace(),o=w.objects[0],s=session(w,o);
  const context=createManualOperationContext({operationId:42,type:"save",workspace:w,object:o,session:s});
  w.workspace_id="workspace-b";
  w.workspace_revision=99;
  o.object_id="object-b";
  o.object_version=99;
  s.generation=99;
  s.identity="changed";
  s.editor.mask[0]=0;
  assert.equal(context.workspaceId,"workspace-a");
  assert.equal(context.objectId,"object-a");
  assert.equal(context.expectedWorkspaceRevision,7);
  assert.equal(context.expectedObjectVersion,4);
  assert.equal(context.expectedManualRevision,1);
  assert.equal(context.brushGeneration,11);
  assert.equal(context.brushEditRevision,5);
  assert.deepEqual([...context.editedMask],[1,1,0,1]);
});

test("manual operation registry allows only one object operation and invalidates by scope",()=>{
  const registry=new ManualOperationRegistry();
  const context=createManualOperationContext({type:"save",workspace:workspace(),object:object(),session:session()});
  const started=registry.begin(context);
  assert.equal(started.operationId,1);
  assert.equal(registry.begin({...context,type:"clear"}),null);
  assert.equal(registry.isCurrent(started),true);
  registry.invalidateObject("workspace-a","object-a");
  assert.equal(registry.isCurrent(started),false);
  const second=registry.begin({...context,type:"clear"});
  assert.equal(registry.has("workspace-a","object-a"),true);
  registry.invalidateWorkspace("workspace-a");
  assert.equal(registry.isCurrent(second),false);
});

test("operation matching rejects stale workspace, object, revision, and session changes",()=>{
  const w=workspace(),o=w.objects[0],s=session(w,o);
  const context=createManualOperationContext({operationId:2,type:"save",workspace:w,object:o,session:s});
  assert.equal(manualOperationMatchesCurrentState(context,{workspace:w,object:o,session:s}),true);
  assert.equal(manualOperationMatchesCurrentState(context,{workspace:{...w,workspace_revision:8},object:o,session:s}),false);
  assert.equal(manualOperationMatchesCurrentState(context,{workspace:w,object:{...o,object_version:5},session:s}),false);
  assert.equal(manualOperationMatchesCurrentState(context,{workspace:w,object:o,session:{...s,generation:12}}),false);
  assert.equal(manualOperationMatchesCurrentState(context,{workspace:{...w,workspace_id:"workspace-b"},object:o,session:s}),false);
});

test("manual conflict compatibility distinguishes structural changes from base changes",()=>{
  const w=workspace(),o=w.objects[0],s=session(w,o);
  const context=createManualOperationContext({operationId:3,type:"save",workspace:w,object:o,session:s});
  const structural=workspace({workspace_revision:8,objects:[object({object_version:5})]});
  assert.equal(manualConflictCompatibility(context,structural).compatible,true);
  const manualChanged=workspace({objects:[object({manual_mask:{...o.manual_mask,manual_revision:2}})]});
  assert.equal(manualConflictCompatibility(context,manualChanged).reason,"manual_revision_changed");
  const samChanged=workspace({objects:[object({sam_draft:{...o.sam_draft,prompt_revision:4}})]});
  assert.equal(manualConflictCompatibility(context,samChanged).reason,"sam_prompt_changed");
  const candidateChanged=workspace({objects:[object({sam_draft:{...o.sam_draft,selected_candidate_index:1}})]});
  assert.equal(manualConflictCompatibility(context,candidateChanged).reason,"selected_candidate_changed");
  assert.equal(manualConflictCompatibility(context,workspace({objects:[]})).reason,"object_deleted");
});

test("manual operations require current editable state and no active work",()=>{
  const w=workspace(),o=w.objects[0];
  const s=createBrushSession({workspace:w,object:o,sourceWidth:2,sourceHeight:2,mask:new Uint8Array([0,0,0,0]),generation:11});
  assert.equal(canStartManualOperation({workspace:w,object:o,session:s,type:"save"}),false);
  s.editor.paint({x:0,y:0},{x:0,y:0},1);
  s.editor.commit();
  assert.equal(canStartManualOperation({workspace:w,object:o,session:s,type:"save"}),true);
  assert.equal(canStartManualOperation({workspace:w,object:o,session:s,type:"save",operationActive:true}),false);
  assert.equal(canStartManualOperation({workspace:w,object:o,session:s,type:"save",predictionActive:true}),false);
  assert.equal(canStartManualOperation({workspace:w,object:o,session:s,type:"save",structuralBusy:true}),false);
  assert.equal(canStartManualOperation({workspace:w,object:o,session:null,type:"clear"}),true);
  assert.equal(canStartManualOperation({workspace:w,object:{...o,manual_mask:{manual_revision:0}},session:null,type:"clear"}),false);
});
