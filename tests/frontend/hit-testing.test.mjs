import test from "node:test";
import assert from "node:assert/strict";
import {HitCycle,hitTestMasks} from "../../frontend/mask-utils.js";

test("hit testing uses mask pixels and ignores bounding-box-only hits",()=>{
  const objects=[{object_id:"b",maskData:new Uint8Array([0,0,0,1])},{object_id:"a",maskData:new Uint8Array([0,0,0,1])},{object_id:"box-only",maskData:new Uint8Array([0,0,0,0])}];
  assert.deepEqual(hitTestMasks(objects,0,0,2,new Map()),[]);
  assert.deepEqual(hitTestMasks(objects,1,1,2,new Map()),["a","b"]);
});

test("overlap cycling is stable and deterministic",()=>{
  const cycle=new HitCycle(),ids=["a","b","c"];
  assert.equal(cycle.pick(ids,4,9),"a");
  assert.equal(cycle.pick(ids,4,9),"b");
  assert.equal(cycle.pick(ids,4,9),"c");
  assert.equal(cycle.pick(ids,4,9),"a");
  assert.equal(cycle.pick(ids,5,9),"a");
});

test("hidden objects are excluded from selection",()=>{
  const objects=[{object_id:"a",maskData:new Uint8Array([1])},{object_id:"b",maskData:new Uint8Array([1])}];
  assert.deepEqual(hitTestMasks(objects,0,0,1,new Map([["a",false],["b",true]])),["b"]);
});
