import test from "node:test";
import assert from "node:assert/strict";
import {
  SegmentStore,
  createPromptState,
  mergeCandidateSelectionResponse,
  mergeClearDraftResponse,
  mergePredictionResponse,
  semanticLabelFromDisplayName
} from "../../frontend/segment-state.js";

function workspace(){
  return {
    workspace_id:"workspace",
    workspace_revision:2,
    objects:[
      {object_id:"one",object_version:1,semantic_label:"chair",display_name:"Chair",sam_draft:{prompt_revision:0,points:[],box:null,candidates:[],selected_candidate_index:null}},
      {object_id:"two",object_version:4,semantic_label:"desk",display_name:"Desk",sam_draft:{prompt_revision:3,points:[{x:1,y:2,label:1}],box:null,candidates:[{candidate_index:0,score:.5,area_pixels:12,bbox_xyxy:[0,0,2,3],mask_url:"/api/segmentation-artifacts/sam-candidates/w/two/3/0"}],selected_candidate_index:0}}
    ]
  };
}

test("state load, selection, and semantic label suggestion",()=>{
  const store=new SegmentStore();
  store.load(workspace());
  assert.equal(store.selectedObjectId,"one");
  store.select("two");
  assert.equal(store.selected.object_id,"two");
  assert.equal(semanticLabelFromDisplayName("Office Chair!"),"office_chair");
});

test("prediction response updates only the matching object and preserves IDs",()=>{
  const merged=mergePredictionResponse(workspace(),{
    workspace_revision:5,
    object_version:2,
    object_id:"one",
    prompt_revision:1,
    points:[{x:4,y:5,label:0}],
    box:null,
    candidates:[{candidate_index:0,score:.9,area_pixels:3,bbox_xyxy:[4,5,6,7],mask_url:"/api/segmentation-artifacts/sam-candidates/w/one/1/0"}],
    selected_candidate_index:0
  });
  assert.equal(merged.workspace_revision,5);
  assert.equal(merged.objects[0].object_id,"one");
  assert.equal(merged.objects[0].object_version,2);
  assert.equal(merged.objects[0].sam_draft.points[0].label,0);
  assert.equal(merged.objects[1].object_version,4);
  assert.equal(merged.objects[1].sam_draft.prompt_revision,3);
});

test("candidate selection and clear-draft focused responses merge safely",()=>{
  const selected=mergeCandidateSelectionResponse(workspace(),{workspace_revision:6,object_version:5,object_id:"two",prompt_revision:3,selected_candidate_index:0});
  assert.equal(selected.objects[1].object_version,5);
  assert.equal(selected.objects[1].sam_draft.points.length,1);
  const cleared=mergeClearDraftResponse(selected,{workspace_revision:7,object_version:6,object_id:"two",sam_draft:{prompt_revision:0,points:[],box:null,candidates:[],selected_candidate_index:null}});
  assert.equal(cleared.objects[1].sam_draft.prompt_revision,0);
  assert.equal(cleared.objects[0].object_id,"one");
});

test("prompt revisions advance and failed revisions are not reused",()=>{
  const state=createPromptState(workspace().objects[1]);
  assert.equal(state.nextRevision,4);
  const store=new SegmentStore();
  store.load(workspace());
  store.select("two");
  const first=store.addPoint("two",{x:2,y:2,label:1});
  const second=store.undoPoint("two");
  assert.equal(first.revision,4);
  assert.equal(second.revision,5);
  store.resetPromptState("two");
  assert.equal(store.getPromptState("two").nextRevision,4);
});
