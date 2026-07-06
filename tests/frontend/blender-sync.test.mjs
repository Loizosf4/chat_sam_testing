import test from "node:test";
import assert from "node:assert/strict";
import {BlenderSyncClient,BlenderSyncError,findDuplicateIds,sceneForSync,staleObjectIds} from "../../frontend/blender-sync.js";

if(!globalThis.CustomEvent)globalThis.CustomEvent=class CustomEvent extends Event{constructor(type,options={}){super(type);this.detail=options.detail}};

const object={object_id:"a".repeat(32),version:2,semantic_label:"desk",display_name:"Desk",mask_revision:"b".repeat(32),center:[1,2,3],dimensions:[4,5,6],quaternion:[1,0,0,0]};
function scene(){return{schema_version:"1.0.0",package_revision:7,scene_id:"office",structural_room_proxies:[],semantic_objects:[{...object}],mask_revisions:[{revision_id:object.mask_revision,geometry_invalidation_status:"valid"}]}}
function ok(body){return{ok:true,status:200,json:async()=>body}}
const ack={type:"sync_ack",payload:{}};

test("scene synchronization sends stable IDs, revisions, and handshake",async()=>{const calls=[];const timer={setInterval:()=>1,clearInterval:()=>{}};const client=new BlenderSyncClient({timer,fetchImpl:async(url,options)=>{calls.push({url,options});return url.includes("/events")?ok({latest_sequence:0,events:[]}):ok(ack)}});await client.connect(scene());assert.equal(JSON.parse(calls[0].options.body).type,"handshake");const sync=JSON.parse(calls[1].options.body);assert.equal(sync.type,"scene_sync");assert.equal(sync.object_id,null);assert.equal(sync.payload.scene_package.semantic_objects[0].object_id,object.object_id);assert.equal(sync.revision,7);client.disconnect()});

test("selection command carries object and mask revision",async()=>{let sent;const client=new BlenderSyncClient({fetchImpl:async(_url,options)=>{sent=JSON.parse(options.body);return ok(ack)}});client.scene=scene();await client.select(object);assert.equal(sent.type,"select_highlight");assert.equal(sent.object_id,object.object_id);assert.equal(sent.payload.mask_revision,object.mask_revision)});

test("duplicate IDs fail before transport",async()=>{const duplicate=scene();duplicate.structural_room_proxies=[{proxy_id:object.object_id}];let called=false;const client=new BlenderSyncClient({fetchImpl:async()=>{called=true;return ok(ack)}});await assert.rejects(client.connect(duplicate),error=>error.code==="duplicate_semantic_id");assert.equal(called,false);assert.deepEqual(findDuplicateIds(duplicate),[object.object_id])});

test("stale masks are identified and sent with scene sync",()=>{const value=scene();value.mask_revisions[0].geometry_invalidation_status="recompute_required";assert.deepEqual(staleObjectIds(value),[object.object_id]);assert.equal(sceneForSync(value).semantic_objects[0].mask_revision,object.mask_revision)});

test("disconnect and reconnect resumes event polling",async()=>{let interval=0,cleared=0;const timer={setInterval:()=>++interval,clearInterval:()=>cleared++};const fetchImpl=async url=>url.includes("/events")?ok({latest_sequence:0,events:[]}):ok(ack);const client=new BlenderSyncClient({timer,fetchImpl});await client.connect(scene());client.disconnect();await client.connect(scene());assert.equal(interval,2);assert.equal(cleared,1);client.disconnect()});

test("structured stale-revision errors reach the website",async()=>{const client=new BlenderSyncClient({fetchImpl:async()=>({ok:false,status:409,json:async()=>({type:"error",payload:{code:"stale_revision",message:"incoming scene is stale",details:{blender_revision:8}}})})});await assert.rejects(client.command("scene_sync",{scene:scene()}),error=>error instanceof BlenderSyncError&&error.code==="stale_revision"&&error.details.blender_revision===8)});

test("selection and transform events are delivered end to end",async()=>{let poll;const timer={setInterval:fn=>{poll=fn;return 1},clearInterval:()=>{}};let eventsReturned=false;const fetchImpl=async url=>{if(url.includes("/events")){if(eventsReturned)return ok({latest_sequence:2,events:[]});eventsReturned=true;return ok({latest_sequence:2,events:[{type:"selection_changed",object_id:object.object_id},{type:"transform_update",object_id:object.object_id,payload:{transform:{center:[2,2,3],dimensions:[4,5,6],quaternion:[1,0,0,0]}}}]})}return ok(ack)};const client=new BlenderSyncClient({timer,fetchImpl});const received=[];client.addEventListener("selection_changed",e=>received.push(e.detail.type));client.addEventListener("transform_update",e=>received.push(e.detail.type));await client.connect(scene());await new Promise(resolve=>setTimeout(resolve,0));assert.deepEqual(received,["selection_changed","transform_update"]);await poll();client.disconnect()});
