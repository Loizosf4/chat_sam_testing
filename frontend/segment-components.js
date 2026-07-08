const esc=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);

export function objectListHtml(objects,selectedId,localPromptStates=new Map()){
  if(!objects.length)return `<p class="muted-pad">No objects yet</p>`;
  return objects.map(object=>{
    const draft=object.sam_draft||{};
    const points=localPromptStates.get(object.object_id)?.localPoints?.length??draft.points?.length??0;
    const hasMask=(draft.candidates||[]).length>0;
    return `<button type="button" class="seg-object ${object.object_id===selectedId?"is-selected":""}" data-object-id="${esc(object.object_id)}">
      <span><strong>${esc(object.display_name)}</strong><small>${esc(object.semantic_label)}</small></span>
      <span class="object-badges"><em>${points} pts</em><em>${hasMask?"mask":"draft"}</em></span>
    </button>`;
  }).join("");
}

export function candidateListHtml(object,selectedIndex){
  const candidates=object?.sam_draft?.candidates||[];
  if(!candidates.length)return `<p class="muted-pad">No candidates yet</p>`;
  return candidates.map(candidate=>`<button type="button" class="candidate-row ${candidate.candidate_index===selectedIndex?"is-selected":""}" data-candidate-index="${candidate.candidate_index}">
    <span><strong>Candidate ${candidate.candidate_index+1}</strong><small>score ${Number(candidate.score).toFixed(3)}</small></span>
    <span><em>${candidate.area_pixels} px</em><em>[${candidate.bbox_xyxy.join(", ")}]</em></span>
  </button>`).join("");
}
