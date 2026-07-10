import test from "node:test";
import assert from "node:assert/strict";
import {canSelectCandidate,persistCandidateSelectionOptimistic} from "../../frontend/segment-state.js";

test("candidate selection is blocked while prediction is active or pending",()=>{
  assert.equal(canSelectCandidate({running:false,pending:false}),true);
  assert.equal(canSelectCandidate({running:true,pending:false}),false);
  assert.equal(canSelectCandidate({running:false,pending:true}),false);
});

test("failed optimistic candidate selection restores previous selection",async()=>{
  const workspace={workspace_id:"w",workspace_revision:1,objects:[{object_id:"o",object_version:1,sam_draft:{selected_candidate_index:0,candidates:[{candidate_index:0},{candidate_index:1}]}}]};
  await assert.rejects(()=>persistCandidateSelectionOptimistic({workspace,objectId:"o",candidateIndex:1,persist:async()=>{throw new Error("fail")}}));
  assert.equal(workspace.objects[0].sam_draft.selected_candidate_index,0);
});
