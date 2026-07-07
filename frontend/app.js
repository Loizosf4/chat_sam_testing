import {ViewportTransform} from "/static/transform.js";
import {HitCycle,binaryFromImageData,colorForIndex,hitTestMasks,rgbaForMask} from "/static/mask-utils.js";
import {MaskEditor} from "/static/brush.js";
import {ReviewStore} from "/static/state.js";
import {ScenePackageClient} from "/static/api.js";
import {detailsHtml,objectListHtml} from "/static/components.js";
import {BlenderSyncClient} from "/static/blender-sync.js?v=1.0.2";

const $=selector=>document.querySelector(selector);
const canvas=$("#image-canvas"),stage=$("#canvas-stage"),ctx=canvas.getContext("2d"),store=new ReviewStore(),api=new ScenePackageClient(),bridge=new BlenderSyncClient(),view=new ViewportTransform(),cycle=new HitCycle();
let sourceImage=null,editor=null,revisions=[],pointer=null,cursor=null,compare=false,renderQueued=false;
const overlayCache=new WeakMap();

function absoluteUrl(path){return new URL(path,location.origin).href}
function toast(message,error=false){const el=document.createElement("div");el.className=`toast${error?" error":""}`;el.textContent=message;$("#toast-region").append(el);setTimeout(()=>el.remove(),3500)}
function setBusy(message){$("#canvas-empty").textContent=message;$("#canvas-empty").hidden=!message}
function currentObject(){return store.selected}
function selectedRevision(){return store.scene?.mask_revisions.find(r=>r.revision_id===currentObject()?.mask_revision)}

function renderBridgeStatus({status,detail=""}){const el=$("#blender-status"),labels={disconnected:"Disconnected",connecting:"Connecting…",synchronized:"Synchronized",stale:"Stale geometry",error:"Error"};el.dataset.status=status;el.querySelector("output").textContent=labels[status]??status;const banner=$("#sync-error-banner");banner.hidden=status!=="error";$("#sync-error-message").textContent=detail}
async function connectBlender(){if(!store.scene)return;try{await bridge.connect(store.scene)}catch(error){toast(error.message,true)}}
bridge.addEventListener("status",event=>renderBridgeStatus(event.detail));
bridge.addEventListener("selection_changed",async event=>{const message=event.detail;if(message.scene_id!==store.scene?.scene_id||!message.object_id)return;await selectObject(message.object_id,true);await bridge.acknowledge(message,{selected:true}).catch(()=>{})});
bridge.addEventListener("transform_update",async event=>{const message=event.detail;if(message.scene_id!==store.scene?.scene_id)return;const object=store.scene.semantic_objects.find(item=>item.object_id===message.object_id);if(!object)return;try{const raw=await api.updateTransform(store.scene,object,message.payload.transform);await replaceMetadataScene(raw,false);await bridge.sync(store.scene);await bridge.acknowledge(message,{saved_revision:raw.package_revision});toast(`Saved Blender transform for ${object.display_name}`)}catch(error){renderBridgeStatus({status:"error",detail:error.status===409?"Transform rejected: scene or mask revision is stale.":error.message});toast(error.message,true)}});

async function imageElement(url){return new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(new Error(`Could not load ${url}`));img.src=absoluteUrl(url)})}
async function fetchMask(url,width,height){const img=await imageElement(url);const c=document.createElement("canvas");c.width=width;c.height=height;const cctx=c.getContext("2d",{willReadFrequently:true});cctx.drawImage(img,0,0,width,height);return binaryFromImageData(cctx.getImageData(0,0,width,height))}
async function enrichScene(scene,previous=null){const previousById=new Map((previous?.semantic_objects??[]).map(o=>[o.object_id,o]));scene.semantic_objects=await Promise.all(scene.semantic_objects.map(async(object,index)=>{const prior=previousById.get(object.object_id);const maskData=prior?.mask_revision===object.mask_revision&&prior.maskData?prior.maskData:await fetchMask(object.mask_url,scene.image_dimensions.width,scene.image_dimensions.height);return{...object,color:prior?.color??colorForIndex(index),maskData}}));return scene}

async function loadScene(sceneId){try{setBusy("Loading scene package…");bridge.disconnect();const raw=await api.getScene(sceneId);sourceImage=await imageElement(raw.source_image.url);const scene=await enrichScene(raw);view.setImageSize(scene.image_dimensions.width,scene.image_dimensions.height).fit();store.load(scene);$("#scene-id").value=scene.scene_id;history.replaceState(null,"",`?scene=${encodeURIComponent(scene.scene_id)}`);$("#empty-state").hidden=true;$("#review-app").hidden=false;setBusy("");await prepareSelected();resizeCanvas();toast(`Loaded ${scene.semantic_objects.length} semantic objects`);await connectBlender()}catch(error){setBusy("");toast(error.message,true)}}
async function prepareSelected(){const object=currentObject();if(!object){editor=null;revisions=[];renderAll();return}editor=new MaskEditor(object.maskData,store.scene.image_dimensions.width,store.scene.image_dimensions.height);store.maskDirty=false;try{const result=await api.history(store.scene.scene_id,object.object_id);revisions=result.revisions}catch{revisions=store.scene.mask_revisions.filter(r=>r.object_id===object.object_id)}renderAll()}
async function replaceMetadataScene(raw,synchronize=true){const dirty=Boolean(editor?.dirty);store.replaceScene(await enrichScene(raw,store.scene));if(dirty)store.setMaskDirty(true);if(synchronize)await bridge.sync(store.scene).catch(error=>renderBridgeStatus({status:"error",detail:error.message}))}

function visibleCanvasObjects(){const all=store.scene?.semantic_objects??[];return all.filter(o=>store.visibility.get(o.object_id)!==false&&($("#combined-overlay").checked||o.object_id===store.selectedObjectId))}
function overlayCanvas(mask,width,height,color,highlight=false){let byKey=overlayCache.get(mask);const key=`${color}:${highlight}`;if(byKey?.[key])return byKey[key];const c=document.createElement("canvas");c.width=width;c.height=height;c.getContext("2d").putImageData(rgbaForMask(mask,width,height,color,1,highlight),0,0);byKey=byKey??{};byKey[key]=c;overlayCache.set(mask,byKey);return c}
function drawMask(object,mask,alpha,highlight=false,clip=null){const {image_dimensions:d}=store.scene,o=view.origin,s=view.scale;ctx.save();if(clip){ctx.beginPath();ctx.rect(clip.x,0,clip.width,view.viewportHeight);ctx.clip()}ctx.globalAlpha=alpha;ctx.imageSmoothingEnabled=false;ctx.drawImage(overlayCanvas(mask,d.width,d.height,object.color,highlight),o.x,o.y,d.width*s,d.height*s);ctx.restore()}
function draw(){renderQueued=false;const cssWidth=stage.clientWidth,cssHeight=stage.clientHeight;if(!cssWidth||!cssHeight)return;const dpr=Math.max(1,window.devicePixelRatio||1);if(canvas.width!==Math.round(cssWidth*dpr)||canvas.height!==Math.round(cssHeight*dpr)){canvas.width=Math.round(cssWidth*dpr);canvas.height=Math.round(cssHeight*dpr);view.setViewport(cssWidth,cssHeight,dpr)}ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,cssWidth,cssHeight);if(!sourceImage||!store.scene)return;const o=view.origin,s=view.scale,d=store.scene.image_dimensions;ctx.imageSmoothingEnabled=true;ctx.drawImage(sourceImage,o.x,o.y,d.width*s,d.height*s);const opacity=Number($("#overlay-opacity").value);for(const object of visibleCanvasObjects()){const selected=object.object_id===store.selectedObjectId;const mask=selected&&editor?editor.mask:object.maskData;if(compare&&selected&&editor){const split=cssWidth/2;drawMask(object,editor.base,opacity,selected,{x:0,width:split});drawMask(object,editor.mask,opacity,selected,{x:split,width:cssWidth-split})}else drawMask(object,mask,opacity,selected)}if(cursor&&(store.tool==="add"||store.tool==="remove")){ctx.save();ctx.beginPath();ctx.arc(cursor.x,cursor.y,Number($("#brush-size").value)*view.scale,0,Math.PI*2);ctx.strokeStyle=store.tool==="add"?"#7cf2b5":"#ff7d84";ctx.lineWidth=1.5;ctx.setLineDash([4,3]);ctx.stroke();ctx.restore()}}
function requestDraw(){if(!renderQueued){renderQueued=true;requestAnimationFrame(draw)}}
function resizeCanvas(){const rect=stage.getBoundingClientRect();view.setViewport(rect.width,rect.height,window.devicePixelRatio||1);requestDraw()}

function renderList(){if(!store.scene)return;const objects=store.scene.semantic_objects;$("#object-count").textContent=objects.length;$("#object-list").innerHTML=objectListHtml(objects,store.selectedObjectId,store.visibility,$("#object-search").value)}
function renderDetails(){$("#object-details").innerHTML=detailsHtml(currentObject(),revisions,store.labelDraft,store.maskDirty)}
function renderSaveState(){const el=$("#save-state");el.classList.toggle("is-saved",!store.unsaved);el.lastChild.textContent=store.unsaved?"Unsaved changes":"All changes saved";$("#undo").disabled=!editor?.undoStack.length;$("#redo").disabled=!editor?.redoStack.length;$("#reset-mask").disabled=!editor?.dirty}
function renderStaleness(){const revisionsById=new Map((store.scene?.mask_revisions??[]).map(r=>[r.revision_id,r]));const stale=(store.scene?.semantic_objects??[]).some(o=>["invalidated","recompute_required"].includes(revisionsById.get(o.mask_revision)?.geometry_invalidation_status));$("#reconstruction-banner").hidden=!stale}
function renderAll(){renderList();renderDetails();renderSaveState();renderStaleness();requestDraw()}

async function selectObject(id,fromBlender=false){if(id===store.selectedObjectId){if(!fromBlender&&currentObject())await bridge.select(currentObject()).catch(error=>renderBridgeStatus({status:"error",detail:error.message}));return}if(store.maskDirty&&!confirm("Discard unsaved mask edits and select another object?"))return;store.select(id);cycle.reset();await prepareSelected();if(!fromBlender&&currentObject())await bridge.select(currentObject()).catch(error=>renderBridgeStatus({status:"error",detail:error.message}))}
function eventPoint(event){const rect=canvas.getBoundingClientRect();return{x:event.clientX-rect.left,y:event.clientY-rect.top}}
function updateEditorDirty(){store.maskDirty=Boolean(editor?.dirty);renderDetails();renderSaveState();requestDraw()}

stage.addEventListener("pointerdown",event=>{if(!store.scene)return;const p=eventPoint(event);stage.setPointerCapture(event.pointerId);if(store.tool==="pan"||event.button===1||event.altKey){pointer={kind:"pan",last:p};return}if((store.tool==="add"||store.tool==="remove")&&editor){const image=view.canvasToImage(p.x,p.y);if(image.x<0||image.y<0||image.x>=view.imageWidth||image.y>=view.imageHeight)return;editor.begin(store.tool);editor.paint(image,image,Number($("#brush-size").value));pointer={kind:"brush",last:image};updateEditorDirty();return}if(store.tool==="inspect"){const pixel=view.canvasToPixel(p.x,p.y);if(!pixel.inside)return;const ids=hitTestMasks(visibleCanvasObjects(),pixel.x,pixel.y,view.imageWidth,store.visibility);const id=cycle.pick(ids,pixel.x,pixel.y);if(id)selectObject(id)}});
stage.addEventListener("pointermove",event=>{const p=eventPoint(event);cursor=p;if(pointer?.kind==="pan"){view.panBy(p.x-pointer.last.x,p.y-pointer.last.y);pointer.last=p}else if(pointer?.kind==="brush"&&editor){const image=view.canvasToImage(p.x,p.y);editor.paint(pointer.last,image,Number($("#brush-size").value));pointer.last=image;updateEditorDirty()}else requestDraw()});
stage.addEventListener("pointerup",()=>{if(pointer?.kind==="brush"&&editor){editor.commit();updateEditorDirty()}pointer=null});
stage.addEventListener("pointerleave",()=>{cursor=null;requestDraw()});
stage.addEventListener("wheel",event=>{if(!store.scene)return;event.preventDefault();const p=eventPoint(event);view.zoomAt(event.deltaY<0?1.12:.89,p.x,p.y);updateZoom();requestDraw()},{passive:false});

$("#scene-form").addEventListener("submit",event=>{event.preventDefault();loadScene($("#scene-id").value.trim())});
$("#retry-blender").addEventListener("click",connectBlender);
$("#object-search").addEventListener("input",renderList);
$("#object-list").addEventListener("click",event=>{const visibility=event.target.closest("[data-toggle-visibility]");if(visibility){event.stopPropagation();store.toggleVisibility(visibility.dataset.toggleVisibility);return}const row=event.target.closest("[data-select-object]");if(row)selectObject(row.dataset.selectObject)});
$("#object-list").addEventListener("keydown",event=>{if((event.key==="Enter"||event.key===" ")&&event.target.matches("[data-select-object]")){event.preventDefault();selectObject(event.target.dataset.selectObject)}});
$("#object-details").addEventListener("input",event=>{if(event.target.id==="label-input"){store.labelDraft=event.target.value;store.labelDirty=event.target.value.trim()!==currentObject().semantic_label;renderSaveState()}});
$("#object-details").addEventListener("click",async event=>{try{if(event.target.id==="save-label"){const label=store.labelDraft.trim();if(!label)return;const raw=await api.updateLabel(store.scene,currentObject(),label);await replaceMetadataScene(raw);toast("Semantic label saved")}if(event.target.id==="approve-object"||event.target.id==="reject-object"){const status=event.target.id==="approve-object"?"approved":"rejected";const raw=await api.updateApproval(store.scene,currentObject(),status);await replaceMetadataScene(raw);toast(`Object ${status}`)}if(event.target.id==="save-mask")await saveMaskRevision()}catch(error){toast(error.status===409?"The scene changed elsewhere. Reload it before saving.":error.message,true)}});

async function saveMaskRevision(){if(!editor?.dirty)return;const object=currentObject(),stableId=object.object_id;const blob=await maskBlob(editor.mask,editor.width,editor.height);const delta={coordinate_system:"source_image_pixels",width:editor.width,height:editor.height,strokes:editor.undoStack.map((changes,index)=>({sequence:index+1,changed_pixels:changes.length}))};const raw=await api.saveRevision(store.scene,object,blob,delta);const scene=await enrichScene(raw,store.scene);store.replaceScene(scene,stableId);await prepareSelected();toast("New immutable mask revision saved")}
function maskBlob(mask,width,height){const c=document.createElement("canvas");c.width=width;c.height=height;const image=c.getContext("2d").createImageData(width,height);for(let i=0;i<mask.length;i++){const v=mask[i]?255:0;image.data[i*4]=v;image.data[i*4+1]=v;image.data[i*4+2]=v;image.data[i*4+3]=255}c.getContext("2d").putImageData(image,0,0);return new Promise(resolve=>c.toBlob(resolve,"image/png"))}

document.querySelectorAll("[data-tool]").forEach(button=>button.addEventListener("click",()=>{store.setTool(button.dataset.tool);document.querySelectorAll("[data-tool]").forEach(b=>b.classList.toggle("is-active",b===button));stage.dataset.tool=button.dataset.tool}));
$("#brush-size").addEventListener("input",()=>{$("#brush-size-output").textContent=`${$("#brush-size").value} px`;requestDraw()});
$("#overlay-opacity").addEventListener("input",()=>{$("#opacity-output").textContent=`${Math.round($("#overlay-opacity").value*100)}%`;requestDraw()});
$("#combined-overlay").addEventListener("change",()=>{cycle.reset();requestDraw()});
$("#undo").addEventListener("click",()=>{editor?.undo();updateEditorDirty()});
$("#redo").addEventListener("click",()=>{editor?.redo();updateEditorDirty()});
$("#reset-mask").addEventListener("click",()=>{editor?.reset();updateEditorDirty()});
$("#compare-mask").addEventListener("click",()=>{compare=!compare;$("#compare-mask").classList.toggle("is-active",compare);$("#before-label").hidden=!compare;$("#after-label").hidden=!compare;requestDraw()});
function updateZoom(){$("#zoom-level").textContent=`${Math.round(view.zoom*100)}%`}
$("#zoom-in").addEventListener("click",()=>{view.zoomAt(1.2,view.viewportWidth/2,view.viewportHeight/2);updateZoom();requestDraw()});
$("#zoom-out").addEventListener("click",()=>{view.zoomAt(.8,view.viewportWidth/2,view.viewportHeight/2);updateZoom();requestDraw()});
$("#fit-view").addEventListener("click",()=>{view.fit();updateZoom();requestDraw()});
window.addEventListener("keydown",event=>{if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==="z"&&!event.shiftKey){event.preventDefault();editor?.undo();updateEditorDirty()}else if((event.ctrlKey||event.metaKey)&&((event.key.toLowerCase()==="z"&&event.shiftKey)||event.key.toLowerCase()==="y")){event.preventDefault();editor?.redo();updateEditorDirty()}});
window.addEventListener("beforeunload",event=>{if(store.unsaved){event.preventDefault();event.returnValue=""}});
new ResizeObserver(resizeCanvas).observe(stage);
store.subscribe(renderAll);
stage.dataset.tool="inspect";
const initialScene=new URLSearchParams(location.search).get("scene");if(initialScene){$("#scene-id").value=initialScene;loadScene(initialScene)}
