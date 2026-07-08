import test from "node:test";
import assert from "node:assert/strict";
import {RenderGate} from "../../frontend/segment-state.js";

function state(overrides={}){
  return {
    workspaceId:"w",
    selectedObjectId:"o",
    promptRevision:1,
    selectedCandidateIndex:0,
    maskUrl:"/mask",
    sourceImageId:"image",
    viewport:{zoom:1,panX:0,panY:0,width:500,height:300,scale:1},
    ...overrides
  };
}

test("older slow render is rejected after newer render is requested",()=>{
  const gate=new RenderGate();
  gate.request();
  const first=gate.snapshot(state());
  gate.request();
  const second=gate.snapshot(state({promptRevision:2,maskUrl:"/mask2"}));
  assert.equal(gate.isCurrent(first,state()),false);
  assert.equal(gate.isCurrent(second,state({promptRevision:2,maskUrl:"/mask2"})),true);
});

test("render snapshot becomes stale on object, candidate, workspace, or viewport switch",()=>{
  const gate=new RenderGate();
  gate.request();
  const snapshot=gate.snapshot(state());
  assert.equal(gate.isCurrent(snapshot,state({selectedObjectId:"other"})),false);
  assert.equal(gate.isCurrent(snapshot,state({selectedCandidateIndex:1})),false);
  assert.equal(gate.isCurrent(snapshot,state({workspaceId:"other"})),false);
  assert.equal(gate.isCurrent(snapshot,state({viewport:{zoom:2,panX:0,panY:0,width:500,height:300,scale:2}})),false);
});

test("current render snapshot remains valid when state is unchanged",()=>{
  const gate=new RenderGate();
  gate.request();
  const snapshot=gate.snapshot(state());
  assert.equal(gate.isCurrent(snapshot,state()),true);
});
