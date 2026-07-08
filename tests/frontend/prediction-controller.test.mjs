import test from "node:test";
import assert from "node:assert/strict";
import {PredictionController} from "../../frontend/segment-state.js";

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
