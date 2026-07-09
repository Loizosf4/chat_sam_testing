import test from "node:test";
import assert from "node:assert/strict";
import {SegmentationWorkspaceClient} from "../../frontend/segment-api.js";
import {canSelectCandidateForObject,canUseSamPrompting} from "../../frontend/segment-state.js";

function object(manualRevision=0){
  return {
    object_id:"object",
    object_version:4,
    manual_mask:{manual_revision:manualRevision},
    sam_draft:{prompt_revision:3,selected_candidate_index:0,candidates:[{candidate_index:0}]}
  };
}

test("manual mask API sends multipart optimistic-lock fields without base64",async()=>{
  let request;
  global.fetch=async(url,options)=>{request={url,options};return{ok:true,json:async()=>({manual_mask:{manual_revision:2}})}};
  const api=new SegmentationWorkspaceClient();
  await api.saveManualMask(
    {workspace_id:"workspace",workspace_revision:8},
    object(1),
    {basePromptRevision:3,baseCandidateIndex:0,manualRevision:1},
    new Blob(["png"],{type:"image/png"})
  );
  assert.equal(request.url,"/api/segmentation-workspaces/workspace/objects/object/manual-mask");
  assert.equal(request.options.method,"PUT");
  assert.equal(request.options.body.get("base_prompt_revision"),"3");
  assert.equal(request.options.body.get("base_candidate_index"),"0");
  assert.equal(request.options.body.get("expected_workspace_revision"),"8");
  assert.equal(request.options.body.get("expected_object_version"),"4");
  assert.equal(request.options.body.get("expected_manual_revision"),"1");
  assert.equal(request.options.body.get("edited_mask") instanceof Blob,true);
  assert.equal(String(request.options.body.get("edited_mask")).includes("base64"),false);
});

test("manual mask clear API sends revision query fields",async()=>{
  let request;
  global.fetch=async(url,options)=>{request={url,options};return{ok:true,json:async()=>({})}};
  const api=new SegmentationWorkspaceClient();
  await api.clearManualMask({workspace_id:"workspace",workspace_revision:9},object(2));
  assert.equal(request.options.method,"DELETE");
  assert.equal(request.url,"/api/segmentation-workspaces/workspace/objects/object/manual-mask?expected_workspace_revision=9&expected_object_version=4&expected_manual_revision=2");
});

test("saved or dirty manual work blocks SAM prompting and candidate changes",()=>{
  assert.equal(canUseSamPrompting({object:object(1),promptState:{running:false,pending:false},samReady:true}),false);
  assert.equal(canUseSamPrompting({object:object(0),promptState:{running:false,pending:false},samReady:true,brushDirty:true}),false);
  assert.equal(canUseSamPrompting({object:object(0),promptState:{running:false,pending:false},samReady:true}),true);
  assert.equal(canSelectCandidateForObject({object:object(1),promptState:{running:false,pending:false}}),false);
  assert.equal(canSelectCandidateForObject({object:object(0),promptState:{running:false,pending:false},brushDirty:true}),false);
  assert.equal(canSelectCandidateForObject({object:object(0),promptState:{running:false,pending:false}}),true);
});
