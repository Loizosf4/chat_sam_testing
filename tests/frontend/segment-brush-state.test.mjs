import test from "node:test";
import assert from "node:assert/strict";
import {
  MANUAL_BASE_MISMATCH_MESSAGE,
  NO_SAM_MASK_MESSAGE,
  binaryMaskRgba,
  brushBaseDescriptor,
  brushIdentity,
  brushSaveState,
  brushSessionMatches,
  createBrushSession,
  markBrushEdited,
  rebaseBrushSession
} from "../../frontend/segment-brush.js";

function object(overrides={}){
  return {
    object_id:"object",
    object_version:2,
    sam_draft:{
      prompt_revision:3,
      points:[{x:1,y:1,label:1}],
      candidates:[{candidate_index:0,mask_url:"/sam/0"},{candidate_index:1,mask_url:"/sam/1"}],
      selected_candidate_index:0
    },
    manual_mask:{manual_revision:0,base_prompt_revision:null,base_candidate_index:null,composite_mask_url:null},
    ...overrides
  };
}
const workspace={workspace_id:"workspace",image_dimensions:{width:4,height:3}};

test("effective brush base selects candidate, manual composite, or no editor",()=>{
  assert.equal(brushBaseDescriptor(workspace,object({sam_draft:{prompt_revision:0,candidates:[],selected_candidate_index:null}})).error,NO_SAM_MASK_MESSAGE);
  const candidate=brushBaseDescriptor(workspace,object());
  assert.equal(candidate.maskType,"sam-candidate");
  assert.equal(candidate.sourceMaskUrl,"/sam/0");
  assert.equal(candidate.basePromptRevision,3);
  const manual=brushBaseDescriptor(workspace,object({manual_mask:{manual_revision:2,base_prompt_revision:3,base_candidate_index:0,composite_mask_url:"/manual/2"}}));
  assert.equal(manual.maskType,"manual-composite");
  assert.equal(manual.sourceMaskUrl,"/manual/2");
  assert.equal(manual.manualRevision,2);
  assert.equal(brushBaseDescriptor(workspace,object({manual_mask:{manual_revision:2,base_prompt_revision:4,base_candidate_index:0,composite_mask_url:"/manual/2"}})).error,MANUAL_BASE_MISMATCH_MESSAGE);
});

test("brush session identity changes with manual revision and rejects stale generation",()=>{
  const mask=new Uint8Array(12);
  const session=createBrushSession({workspace,object:object(),sourceWidth:4,sourceHeight:3,mask,generation:1});
  assert.equal(session.editor.mask.length,12);
  assert.equal(brushSessionMatches(session,workspace,object(),1),true);
  assert.equal(brushSessionMatches(session,workspace,object(),2),false);
  const first=brushIdentity({workspaceId:"w",objectId:"o",maskType:"manual",basePromptRevision:1,baseCandidateIndex:0,manualRevision:1,sourceMaskUrl:"/m1"});
  const second=brushIdentity({workspaceId:"w",objectId:"o",maskType:"manual",basePromptRevision:1,baseCandidateIndex:0,manualRevision:2,sourceMaskUrl:"/m2"});
  assert.notEqual(first,second);
});

test("brush editing, reset, and rebase track dirty state",()=>{
  const base=new Uint8Array(9);
  const session=createBrushSession({workspace,object:object(),sourceWidth:3,sourceHeight:3,mask:base,generation:1});
  session.editor.begin("add");
  session.editor.paint({x:1,y:1},{x:1,y:1},0.6);
  session.editor.commit();
  markBrushEdited(session);
  assert.equal(session.editor.dirty,true);
  assert.equal(session.editor.undoStack.length,1);
  assert.equal(session.editor.undo(),true);
  assert.equal(session.editor.redo(),true);
  session.editor.reset();
  markBrushEdited(session);
  assert.equal(session.editor.dirty,false);
  session.editor.begin("add");
  session.editor.paint({x:1,y:1},{x:1,y:1},0.6);
  session.editor.commit();
  rebaseBrushSession(session);
  assert.equal(session.editor.dirty,false);
  assert.equal(session.editor.undoStack.length,0);
  assert.equal(brushSaveState(session),"Saved");
});

test("binary mask encoder produces full-resolution black and white pixels",()=>{
  const rgba=binaryMaskRgba(new Uint8Array([0,1]),2,1);
  assert.deepEqual([...rgba],[0,0,0,255,255,255,255,255]);
});
