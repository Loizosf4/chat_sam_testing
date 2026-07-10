import test from "node:test";
import assert from "node:assert/strict";
import {WorkspaceLoadGate} from "../../frontend/segment-state.js";

test("workspace load generation lets only newest load apply",()=>{
  const gate=new WorkspaceLoadGate();
  const loadA=gate.begin();
  const loadB=gate.begin();
  assert.equal(gate.isCurrent(loadA),false);
  assert.equal(gate.isCurrent(loadB),true);
});

test("stale SAM preparation result cannot alter newer load status",async()=>{
  const gate=new WorkspaceLoadGate();
  const statuses=[];
  const loadA=gate.begin();
  const loadB=gate.begin();
  function applyStatus(generation,status){if(gate.isCurrent(generation))statuses.push(status)}
  applyStatus(loadB,"b-ready");
  applyStatus(loadA,"a-ready");
  assert.deepEqual(statuses,["b-ready"]);
});
