import test from "node:test";
import assert from "node:assert/strict";
import {CandidateMaskCache,ControllerRegistry,PredictionController,candidateCacheKey,predictionStatusVisible} from "../../frontend/segment-state.js";

test("failed reset can dispose and recreate a functional controller",async()=>{
  const registry=new ControllerRegistry();
  const submitted=[];
  const first=new PredictionController({workspaceId:"w",objectId:"o",submit:s=>{submitted.push(["first",s]);return Promise.resolve({object_id:"o",prompt_revision:s.revision})},apply:()=>{}});
  registry.set("w","o",first);
  registry.dispose("w","o");
  first.enqueue({revision:1});
  const recreated=new PredictionController({workspaceId:"w",objectId:"o",submit:s=>{submitted.push(["recreated",s]);return Promise.resolve({object_id:"o",prompt_revision:s.revision})},apply:()=>{}});
  registry.set("w","o",recreated);
  recreated.enqueue({revision:2});
  await Promise.resolve();
  assert.deepEqual(submitted.map(item=>item[0]),["recreated"]);
});

test("failed deletion does not leave surviving object with unusable controller",async()=>{
  const registry=new ControllerRegistry();
  registry.set("w","o",new PredictionController({workspaceId:"w",objectId:"o",submit:async s=>({object_id:"o",prompt_revision:s.revision}),apply:()=>{}}));
  registry.dispose("w","o");
  const replacement=new PredictionController({workspaceId:"w",objectId:"o",submit:async s=>({object_id:"o",prompt_revision:s.revision}),apply:()=>{}});
  registry.set("w","o",replacement);
  replacement.enqueue({revision:3});
  await Promise.resolve();
  assert.equal(replacement.submitted.length,1);
  assert.equal(replacement.submitted[0].revision,3);
});

test("candidate cache is scoped and cleared by object or workspace replacement",()=>{
  const cache=new CandidateMaskCache();
  const one=candidateCacheKey({workspaceId:"w1",objectId:"o1",promptRevision:1,candidateIndex:0,maskUrl:"/m"});
  const two=candidateCacheKey({workspaceId:"w1",objectId:"o1",promptRevision:2,candidateIndex:0,maskUrl:"/m"});
  const other=candidateCacheKey({workspaceId:"w1",objectId:"o2",promptRevision:1,candidateIndex:0,maskUrl:"/m"});
  cache.set(one,"old");
  cache.set(two,"new");
  cache.set(other,"other");
  assert.notEqual(one,two);
  cache.clearObject("w1","o1");
  assert.equal(cache.get(one),undefined);
  assert.equal(cache.get(two),undefined);
  assert.equal(cache.get(other),"other");
  cache.clear();
  assert.equal(cache.get(other),undefined);
});

test("updating indicator is derived from selected object state only",()=>{
  assert.equal(predictionStatusVisible({controller:{running:true,pending:false}}),true);
  assert.equal(predictionStatusVisible({controller:{running:false,pending:true}}),true);
  assert.equal(predictionStatusVisible({promptState:{running:false,pending:true}}),true);
  assert.equal(predictionStatusVisible({controller:{running:false,pending:false},promptState:{running:false,pending:false}}),false);
});
