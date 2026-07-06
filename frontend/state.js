export class ReviewStore {
  constructor(){this.scene=null;this.selectedObjectId=null;this.visibility=new Map();this.tool="inspect";this.labelDraft="";this.labelDirty=false;this.maskDirty=false;this.listeners=new Set()}
  subscribe(listener){this.listeners.add(listener);return()=>this.listeners.delete(listener)}
  emit(){this.listeners.forEach(fn=>fn(this))}
  load(scene){this.scene=scene;this.selectedObjectId=scene.semantic_objects[0]?.object_id??null;this.visibility=new Map(scene.semantic_objects.map(o=>[o.object_id,true]));this.syncDraft();this.maskDirty=false;this.emit()}
  get selected(){return this.scene?.semantic_objects.find(o=>o.object_id===this.selectedObjectId)??null}
  select(id){if(this.scene?.semantic_objects.some(o=>o.object_id===id)){this.selectedObjectId=id;this.syncDraft();this.maskDirty=false;this.emit()}}
  syncDraft(){this.labelDraft=this.selected?.semantic_label??"";this.labelDirty=false}
  setLabel(value){this.labelDraft=value;this.labelDirty=value.trim()!==(this.selected?.semantic_label??"");this.emit()}
  setMaskDirty(value){this.maskDirty=value;this.emit()}
  toggleVisibility(id){this.visibility.set(id,this.visibility.get(id)===false);this.emit()}
  setTool(tool){this.tool=tool;this.emit()}
  replaceScene(scene,preserveId=this.selectedObjectId){this.scene=scene;this.selectedObjectId=scene.semantic_objects.some(o=>o.object_id===preserveId)?preserveId:scene.semantic_objects[0]?.object_id??null;for(const o of scene.semantic_objects)if(!this.visibility.has(o.object_id))this.visibility.set(o.object_id,true);this.syncDraft();this.maskDirty=false;this.emit()}
  get unsaved(){return this.labelDirty||this.maskDirty}
}
