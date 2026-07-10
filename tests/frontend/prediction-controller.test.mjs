import test from "node:test";
import assert from "node:assert/strict";
import {ControllerRegistry,PredictionController,controllerKey} from "../../frontend/segment-state.js";

function deferred(){
  let resolve,reject;
  const promise=new Promise((res,rej)=>{resolve=res;reject=rej});
  return {promise,resolve,reject};
}

test("latest-state-wins submits first and newest pending state only",async()=>{
  const requests=[];
  const first=deferred(),second=deferred();
  const applied=[];
  const controller=new PredictionController({
    objectId:"object",
    submit:snapshot=>{requests.push(snapshot);return requests.length===1?first.promise:second.promise},
    apply:response=>applied.push(response)
  });
  controller.enqueue({revision:1,points:[1]});
  controller.enqueue({revision:2,points:[1,2]});
  controller.enqueue({revision:3,points:[1,2,3]});
  assert.equal(requests.length,1);
  first.resolve({object_id:"object",prompt_revision:1});
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(applied.length,0);
  assert.equal(requests.length,2);
  assert.equal(requests[1].revision,3);
  second.resolve({object_id:"object",prompt_revision:3});
  await Promise.resolve();
  await Promise.resolve();
  assert.deepEqual(applied.map(item=>item.prompt_revision),[3]);
});

test("controller registry disposes invalidated controllers and allows recreation",async()=>{
  const registry=new ControllerRegistry();
  const submitted=[];
  const oldController=new PredictionController({workspaceId:"w",objectId:"o",submit:s=>{submitted.push(["old",s]);return Promise.resolve({object_id:"o",prompt_revision:s.revision})},apply:()=>{}});
  registry.set("w","o",oldController);
  registry.dispose("w","o");
  oldController.enqueue({revision:1});
  assert.equal(submitted.length,0);
  assert.equal(registry.has("w","o"),false);

  const newController=new PredictionController({workspaceId:"w",objectId:"o",submit:s=>{submitted.push(["new",s]);return Promise.resolve({object_id:"o",prompt_revision:s.revision})},apply:()=>{}});
  registry.set("w","o",newController);
  newController.enqueue({revision:2});
  await Promise.resolve();
  assert.equal(submitted.length,1);
  assert.equal(submitted[0][0],"new");
});

test("controller keys and submissions are scoped to workspace and object",async()=>{
  assert.equal(controllerKey("workspace-a","object"),"workspace-a:object");
  const requests=[];
  const controller=new PredictionController({
    workspaceId:"workspace-a",
    objectId:"object",
    submit:snapshot=>{requests.push(snapshot);return Promise.resolve({object_id:"object",prompt_revision:snapshot.revision})},
    apply:()=>{}
  });
  controller.enqueue({revision:7});
  await Promise.resolve();
  assert.equal(requests[0].workspaceId,"workspace-a");
  assert.equal(requests[0].objectId,"object");
});

test("dispose workspace invalidates only matching workspace controllers",()=>{
  const registry=new ControllerRegistry();
  const a=new PredictionController({workspaceId:"a",objectId:"one",submit:async()=>({}),apply:()=>{}});
  const b=new PredictionController({workspaceId:"b",objectId:"one",submit:async()=>({}),apply:()=>{}});
  registry.set("a","one",a);
  registry.set("b","one",b);
  registry.disposeWorkspace("a");
  assert.equal(a.invalidated,true);
  assert.equal(registry.has("a","one"),false);
  assert.equal(b.invalidated,false);
  assert.equal(registry.has("b","one"),true);
});

test("stale conflicts are ignored when newer state is pending",async()=>{
  const first=deferred(),second=deferred();
  const errors=[];
  const requests=[];
  const controller=new PredictionController({
    objectId:"object",
    submit:snapshot=>{requests.push(snapshot);return requests.length===1?first.promise:second.promise},
    apply:()=>{},
    onError:error=>errors.push(error)
  });
  controller.enqueue({revision:4});
  controller.enqueue({revision:5});
  const stale=new Error("stale");
  stale.status=409;
  first.reject(stale);
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(errors.length,0);
  assert.equal(requests[1].revision,5);
});

test("object switching and deletion invalidate pending responses",async()=>{
  const wait=deferred();
  const applied=[];
  const controller=new PredictionController({
    objectId:"old",
    submit:()=>wait.promise,
    apply:response=>applied.push(response),
    isActive:id=>id==="current"
  });
  controller.enqueue({revision:1});
  wait.resolve({object_id:"old",prompt_revision:1});
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(applied.length,0);
  controller.invalidate();
  controller.enqueue({revision:2});
  assert.equal(controller.pending,null);
});
