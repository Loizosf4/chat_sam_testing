export function brushIndices(width,height,x,y,radius){const out=[];const r=Math.max(.5,radius),minX=Math.max(0,Math.floor(x-r)),maxX=Math.min(width-1,Math.ceil(x+r)),minY=Math.max(0,Math.floor(y-r)),maxY=Math.min(height-1,Math.ceil(y+r));for(let py=minY;py<=maxY;py++)for(let px=minX;px<=maxX;px++)if((px-x)**2+(py-y)**2<=r*r)out.push(py*width+px);return out}
export function strokePoints(from,to,spacing=1){const distance=Math.hypot(to.x-from.x,to.y-from.y);const count=Math.max(1,Math.ceil(distance/Math.max(.5,spacing)));return Array.from({length:count+1},(_,i)=>({x:from.x+(to.x-from.x)*i/count,y:from.y+(to.y-from.y)*i/count}))}
export class MaskEditor {
  constructor(mask,width,height){this.width=width;this.height=height;this.base=new Uint8Array(mask);this.mask=new Uint8Array(mask);this.undoStack=[];this.redoStack=[];this.active=null}
  begin(mode){this.active={mode,value:mode==="add"?1:0,changes:new Map()}}
  paint(from,to,radius){if(!this.active)this.begin("add");for(const point of strokePoints(from,to,Math.max(1,radius*.35)))for(const index of brushIndices(this.width,this.height,point.x,point.y,radius)){if(!this.active.changes.has(index))this.active.changes.set(index,this.mask[index]);this.mask[index]=this.active.value}}
  commit(){if(!this.active)return false;const changes=[...this.active.changes].filter(([i,before])=>before!==this.mask[i]).map(([index,before])=>({index,before,after:this.mask[index]}));this.active=null;if(!changes.length)return false;this.undoStack.push(changes);this.redoStack=[];return true}
  undo(){const c=this.undoStack.pop();if(!c)return false;c.forEach(v=>this.mask[v.index]=v.before);this.redoStack.push(c);return true}
  redo(){const c=this.redoStack.pop();if(!c)return false;c.forEach(v=>this.mask[v.index]=v.after);this.undoStack.push(c);return true}
  reset(){this.mask.set(this.base);this.undoStack=[];this.redoStack=[];this.active=null}
  get dirty(){for(let i=0;i<this.mask.length;i++)if(this.mask[i]!==this.base[i])return true;return false}
}
