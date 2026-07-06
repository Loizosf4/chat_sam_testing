import test from "node:test";
import assert from "node:assert/strict";
import {ReviewStore} from "../../frontend/state.js";

const scene={semantic_objects:[{object_id:"one",semantic_label:"desk"},{object_id:"two",semantic_label:"chair"}]};
test("list and canvas selection share one selected object ID",()=>{const store=new ReviewStore();store.load(scene);assert.equal(store.selected.object_id,"one");store.select("two");assert.equal(store.selectedObjectId,"two");assert.equal(store.selected.semantic_label,"chair")});
test("label and mask edits drive unsaved state",()=>{const store=new ReviewStore();store.load(scene);store.setLabel("table");assert.equal(store.unsaved,true);store.syncDraft();store.setMaskDirty(true);assert.equal(store.unsaved,true);store.setMaskDirty(false);assert.equal(store.unsaved,false)});
test("scene refresh preserves stable semantic object selection",()=>{const store=new ReviewStore();store.load(scene);store.select("two");store.replaceScene({semantic_objects:[{object_id:"one",semantic_label:"desk2"},{object_id:"two",semantic_label:"seat"}]});assert.equal(store.selectedObjectId,"two")});
