import test from "node:test";
import assert from "node:assert/strict";
import {ViewportTransform} from "../../frontend/transform.js";
import {canvasPointToImage,fitViewport} from "../../frontend/segment-coordinates.js";

test("screen to image conversion accounts for fit, zoom, and pan",()=>{
  const view=fitViewport(new ViewportTransform(1000,500),800,600,2);
  view.zoomAt(2,400,300).panBy(25,-10);
  const image={x:123.5,y:44.25};
  const canvas=view.imageToCanvas(image.x,image.y);
  const converted=canvasPointToImage(view,canvas.x,canvas.y);
  assert.ok(Math.abs(converted.x-image.x)<1e-9);
  assert.ok(Math.abs(converted.y-image.y)<1e-9);
});

test("clicks outside the image are rejected",()=>{
  const view=fitViewport(new ViewportTransform(100,100),500,300,1);
  assert.equal(canvasPointToImage(view,-1,50),null);
  assert.equal(canvasPointToImage(view,250,150)?.x,50);
});
