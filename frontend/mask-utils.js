export const MASK_COLORS=["#f2a93b","#52b788","#5fa8ff","#ef6f91","#a78bfa","#e879f9","#22c7b8","#f4764f","#84cc5a","#d9b44a"];
export function colorForIndex(index){return MASK_COLORS[index%MASK_COLORS.length]}
export function maskAt(mask,x,y,width){return Boolean(mask?.[y*width+x])}
export function hitTestMasks(objects,x,y,width,visibility=new Map()){return objects.filter(object=>visibility.get(object.object_id)!==false&&maskAt(object.maskData,x,y,width)).map(object=>object.object_id).sort()}
export class HitCycle {
  constructor(){this.key="";this.index=-1}
  pick(ids,x,y){if(!ids.length){this.key="";this.index=-1;return null}const key=`${x}:${y}:${ids.join(",")}`;this.index=key===this.key?(this.index+1)%ids.length:0;this.key=key;return ids[this.index]}
  reset(){this.key="";this.index=-1}
}
export function binaryFromImageData(imageData){const result=new Uint8Array(imageData.width*imageData.height);for(let i=0;i<result.length;i++)result[i]=imageData.data[i*4]>127?1:0;return result}
export function rgbaForMask(mask,width,height,color,alpha=1,highlight=false){const hex=color.replace("#","");const rgb=[0,2,4].map(i=>parseInt(hex.slice(i,i+2),16));const data=new Uint8ClampedArray(width*height*4);for(let y=0;y<height;y++)for(let x=0;x<width;x++){const i=y*width+x;if(!mask[i])continue;const edge=highlight&&(!x||!y||x===width-1||y===height-1||!mask[i-1]||!mask[i+1]||!mask[i-width]||!mask[i+width]);const p=i*4;data[p]=edge?255:rgb[0];data[p+1]=edge?210:rgb[1];data[p+2]=edge?73:rgb[2];data[p+3]=Math.round(255*(edge?1:alpha))}return new ImageData(data,width,height)}
