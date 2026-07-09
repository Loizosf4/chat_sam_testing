import test from "node:test";
import assert from "node:assert/strict";
import {
  SegmentStore,
  createPromptState,
  emptyManualMask,
  manualMaskActive,
  mergeCandidateSelectionResponse,
  mergeClearDraftResponse,
  mergeManualMaskResponse,
  mergePredictionResponse,
  normalizeWorkspace,
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
  assert.deepEqual(store.selected.manual_mask,emptyManualMask());
  store.select("two");
  assert.equal(store.selected.object_id,"two");
  assert.equal(semanticLabelFromDisplayName("Office Chair!"),"office_chair");
});

test("old workspace objects receive empty manual state while active state is preserved",()=>{
  const source=workspace();
  const active={...source,objects:[{...source.objects[0],manual_mask:{manual_revision:2,base_prompt_revision:1,base_candidate_index:0,add_mask_url:"/a",remove_mask_url:"/r",composite_mask_url:"/c",area_pixels:5,bbox_xyxy:[1,2,3,4],updated_at:"now"}},source.objects[1]]};
  const normalized=normalizeWorkspace(active);
  assert.equal(normalized.objects[0].manual_mask.manual_revision,2);
  assert.equal(normalized.objects[1].manual_mask.manual_revision,0);
  assert.equal(manualMaskActive(normalized.objects[0]),true);
  assert.equal(manualMaskActive(normalized.objects[1]),false);
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
  assert.equal(cleared.objects[1].manual_mask.manual_revision,0);
  assert.equal(cleared.objects[0].object_id,"one");
});

test("manual mask focused responses preserve SAM state and stable IDs",()=>{
  const base=workspace();
  const manual={manual_revision:1,base_prompt_revision:3,base_candidate_index:0,add_mask_url:"/api/segmentation-artifacts/manual-masks/w/two/1/add",remove_mask_url:"/api/segmentation-artifacts/manual-masks/w/two/1/remove",composite_mask_url:"/api/segmentation-artifacts/manual-masks/w/two/1/composite",area_pixels:12,bbox_xyxy:[0,0,2,3],updated_at:"now"};
  const merged=mergeManualMaskResponse(base,{workspace_revision:8,object_version:5,object_id:"two",manual_mask:manual});
  assert.equal(merged.workspace_revision,8);
  assert.equal(merged.objects[1].object_id,"two");
  assert.equal(merged.objects[1].object_version,5);
  assert.deepEqual(merged.objects[1].manual_mask,manual);
  assert.equal(merged.objects[1].sam_draft.prompt_revision,3);
  assert.equal(merged.objects[0].object_version,1);
  const cleared=mergeManualMaskResponse(merged,{workspace_revision:9,object_version:6,object_id:"two",manual_mask:emptyManualMask()});
  assert.equal(cleared.objects[1].manual_mask.manual_revision,0);
  assert.equal(cleared.objects[1].sam_draft.candidates.length,1);
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
