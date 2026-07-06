import test from "node:test";
import assert from "node:assert/strict";
import {MaskEditor,brushIndices} from "../../frontend/brush.js";

test("brush clips to exact source-image pixels",()=>{
  assert.deepEqual(brushIndices(3,3,0,0,1).sort((a,b)=>a-b),[0,1,3]);
  const editor=new MaskEditor(new Uint8Array(25),5,5);
  editor.begin("add");editor.paint({x:1,y:2},{x:3,y:2},.6);editor.commit();
  assert.equal(editor.mask[2*5+1],1);assert.equal(editor.mask[2*5+2],1);assert.equal(editor.mask[2*5+3],1);
  assert.equal(editor.mask[1*5+2],0);
});

test("add/remove strokes support undo, redo, and reset to current revision",()=>{
  const base=new Uint8Array(16);base[5]=1;
  const editor=new MaskEditor(base,4,4);
  editor.begin("remove");editor.paint({x:1,y:1},{x:1,y:1},.6);editor.commit();
  assert.equal(editor.mask[5],0);assert.equal(editor.dirty,true);
  assert.equal(editor.undo(),true);assert.equal(editor.mask[5],1);assert.equal(editor.dirty,false);
  assert.equal(editor.redo(),true);assert.equal(editor.mask[5],0);
  editor.reset();assert.deepEqual(editor.mask,base);assert.equal(editor.undo(),false);
});
