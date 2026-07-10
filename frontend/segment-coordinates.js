export function canvasEventPoint(canvas,event){
  const rect=canvas.getBoundingClientRect();
  return {x:event.clientX-rect.left,y:event.clientY-rect.top};
}

export function canvasPointToImage(view,x,y){
  const point=view.canvasToImage(x,y);
  if(point.x<0||point.y<0||point.x>=view.imageWidth||point.y>=view.imageHeight)return null;
  return {x:point.x,y:point.y};
}

export function fitViewport(view,width,height,dpr=1){
  return view.setViewport(width,height,dpr).fit();
}
