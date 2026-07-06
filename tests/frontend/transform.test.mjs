import test from "node:test";
import assert from "node:assert/strict";
import {ViewportTransform} from "../../frontend/transform.js";

test("image and canvas coordinates round-trip under fit, zoom, pan, resize, and DPR",()=>{
  const view=new ViewportTransform(447,447).setViewport(900,600,2);
  view.zoomAt(2.4,321,244).panBy(37,-19);
  for(const point of [[0,0],[123.25,88.75],[446.999,446.999]]){
    const canvas=view.imageToCanvas(...point),image=view.canvasToImage(canvas.x,canvas.y);
    assert.ok(Math.abs(image.x-point[0])<1e-9);
    assert.ok(Math.abs(image.y-point[1])<1e-9);
  }
  view.setViewport(612,777,3);
  const p=view.canvasToPixel(...Object.values(view.imageToCanvas(446.5,12.1)));
  assert.deepEqual(p,{x:446,y:12,inside:true});
});

test("zoom anchoring preserves the source coordinate below the pointer",()=>{
  const view=new ViewportTransform(1000,500).setViewport(800,600,2);
  const before=view.canvasToImage(210,330);
  view.zoomAt(1.8,210,330);
  assert.deepEqual(view.canvasToImage(210,330),before);
});
