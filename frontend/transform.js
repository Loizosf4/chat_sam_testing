export class ViewportTransform {
  constructor(imageWidth=1,imageHeight=1){this.imageWidth=imageWidth;this.imageHeight=imageHeight;this.viewportWidth=1;this.viewportHeight=1;this.dpr=1;this.zoom=1;this.panX=0;this.panY=0}
  setImageSize(width,height){this.imageWidth=width;this.imageHeight=height;return this}
  setViewport(width,height,dpr=1){this.viewportWidth=Math.max(1,width);this.viewportHeight=Math.max(1,height);this.dpr=Math.max(1,dpr);return this}
  get fitScale(){return Math.min(this.viewportWidth/this.imageWidth,this.viewportHeight/this.imageHeight)}
  get scale(){return this.fitScale*this.zoom}
  get origin(){return{x:(this.viewportWidth-this.imageWidth*this.scale)/2+this.panX,y:(this.viewportHeight-this.imageHeight*this.scale)/2+this.panY}}
  imageToCanvas(x,y){const o=this.origin;return{x:o.x+x*this.scale,y:o.y+y*this.scale}}
  canvasToImage(x,y){const o=this.origin;return{x:(x-o.x)/this.scale,y:(y-o.y)/this.scale}}
  canvasToPixel(x,y){const p=this.canvasToImage(x,y);return{x:Math.max(0,Math.min(this.imageWidth-1,Math.floor(p.x))),y:Math.max(0,Math.min(this.imageHeight-1,Math.floor(p.y))),inside:p.x>=0&&p.y>=0&&p.x<this.imageWidth&&p.y<this.imageHeight}}
  zoomAt(factor,canvasX,canvasY){const before=this.canvasToImage(canvasX,canvasY);this.zoom=Math.max(.25,Math.min(16,this.zoom*factor));const after=this.imageToCanvas(before.x,before.y);this.panX+=canvasX-after.x;this.panY+=canvasY-after.y;return this}
  panBy(dx,dy){this.panX+=dx;this.panY+=dy;return this}
  fit(){this.zoom=1;this.panX=0;this.panY=0;return this}
}
