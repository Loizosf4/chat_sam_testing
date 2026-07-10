export class SegmentationWorkspaceClient {
  constructor(base=""){this.base=base.replace(/\/$/,"")}
  async request(path,options={}){
    const response=await fetch(`${this.base}${path}`,options);
    if(!response.ok){
      const body=await response.json().catch(()=>({}));
      const error=new Error(body.detail||`Request failed (${response.status})`);
      error.status=response.status;
      throw error;
    }
    return response.json();
  }
  createWorkspace(file){
    const form=new FormData();
    form.append("source_image",file,file.name);
    return this.request("/api/segmentation-workspaces",{method:"POST",body:form});
  }
  getWorkspace(workspaceId){return this.request(`/api/segmentation-workspaces/${encodeURIComponent(workspaceId)}`)}
  createObject(workspace,semanticLabel,displayName){
    return this.request(`/api/segmentation-workspaces/${encodeURIComponent(workspace.workspace_id)}/objects`,{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({semantic_label:semanticLabel,display_name:displayName,expected_workspace_revision:workspace.workspace_revision})
    });
  }
  updateObject(workspace,object,semanticLabel,displayName){
    return this.request(`/api/segmentation-workspaces/${encodeURIComponent(workspace.workspace_id)}/objects/${object.object_id}`,{
      method:"PATCH",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({semantic_label:semanticLabel,display_name:displayName,expected_workspace_revision:workspace.workspace_revision,expected_object_version:object.object_version})
    });
  }
  deleteObject(workspace,object){
    const query=new URLSearchParams({expected_workspace_revision:workspace.workspace_revision,expected_object_version:object.object_version});
    return this.request(`/api/segmentation-workspaces/${encodeURIComponent(workspace.workspace_id)}/objects/${object.object_id}?${query}`,{method:"DELETE"});
  }
  prepareSam(workspaceId){return this.request(`/api/segmentation-workspaces/${encodeURIComponent(workspaceId)}/prepare-sam`,{method:"POST"})}
  predict(workspaceId,objectId,payload){
    return this.request(`/api/segmentation-workspaces/${encodeURIComponent(workspaceId)}/objects/${objectId}/predict`,{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify(payload)
    });
  }
  selectCandidate(workspace,object,promptRevision,candidateIndex){
    return this.request(`/api/segmentation-workspaces/${encodeURIComponent(workspace.workspace_id)}/objects/${object.object_id}/select-candidate`,{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({prompt_revision:promptRevision,candidate_index:candidateIndex,expected_workspace_revision:workspace.workspace_revision,expected_object_version:object.object_version})
    });
  }
  clearDraft(workspace,object){
    const query=new URLSearchParams({expected_workspace_revision:workspace.workspace_revision,expected_object_version:object.object_version});
    return this.request(`/api/segmentation-workspaces/${encodeURIComponent(workspace.workspace_id)}/objects/${object.object_id}/sam-draft?${query}`,{method:"DELETE"});
  }
  saveManualMask({
    workspaceId,
    objectId,
    expectedWorkspaceRevision,
    expectedObjectVersion,
    expectedManualRevision,
    basePromptRevision,
    baseCandidateIndex,
    blob
  }){
    const form=new FormData();
    form.append("edited_mask",blob,"edited-mask.png");
    form.append("base_prompt_revision",String(basePromptRevision));
    form.append("base_candidate_index",String(baseCandidateIndex));
    form.append("expected_workspace_revision",String(expectedWorkspaceRevision));
    form.append("expected_object_version",String(expectedObjectVersion));
    form.append("expected_manual_revision",String(expectedManualRevision));
    return this.request(`/api/segmentation-workspaces/${encodeURIComponent(workspaceId)}/objects/${objectId}/manual-mask`,{method:"PUT",body:form});
  }
  clearManualMask({workspaceId,objectId,expectedWorkspaceRevision,expectedObjectVersion,expectedManualRevision}){
    const query=new URLSearchParams({expected_workspace_revision:expectedWorkspaceRevision,expected_object_version:expectedObjectVersion,expected_manual_revision:expectedManualRevision});
    return this.request(`/api/segmentation-workspaces/${encodeURIComponent(workspaceId)}/objects/${objectId}/manual-mask?${query}`,{method:"DELETE"});
  }
  createExport({workspaceId,expectedWorkspaceRevision,expectedObjects}){
    return this.request(`/api/segmentation-workspaces/${encodeURIComponent(workspaceId)}/exports`,{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({expected_workspace_revision:expectedWorkspaceRevision,expected_objects:expectedObjects.map(item=>({...item})),include_previews:true})
    });
  }
  listExports(workspaceId){
    return this.request(`/api/segmentation-workspaces/${encodeURIComponent(workspaceId)}/exports`);
  }
  getExport(workspaceId,exportId){
    return this.request(`/api/segmentation-workspaces/${encodeURIComponent(workspaceId)}/exports/${encodeURIComponent(exportId)}`);
  }
  getJsonArtifact(url){
    if(!String(url||"").startsWith("/api/segmentation-artifacts/"))throw new Error("Artifact URL must be an application URL");
    return this.request(url);
  }
  getReconstructionHealth(){
    return this.request("/api/segmentation-reconstruction/health");
  }
  startReconstruction({
    workspaceId,
    exportId,
    expectedWorkspaceRevision,
    expectedExportArchiveSha256,
    resolutionLevel,
    numTokens
  }){
    const payload={
      expected_workspace_revision:expectedWorkspaceRevision,
      expected_export_archive_sha256:expectedExportArchiveSha256,
      resolution_level:resolutionLevel,
      num_tokens:numTokens===undefined?null:numTokens
    };
    return this.request(`/api/segmentation-workspaces/${encodeURIComponent(workspaceId)}/exports/${encodeURIComponent(exportId)}/reconstructions`,{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify(payload)
    });
  }
  listReconstructions(workspaceId){
    return this.request(`/api/segmentation-workspaces/${encodeURIComponent(workspaceId)}/reconstructions`);
  }
  getReconstruction(workspaceId,jobId){
    return this.request(`/api/segmentation-workspaces/${encodeURIComponent(workspaceId)}/reconstructions/${encodeURIComponent(jobId)}`);
  }
  getReviewSceneStatus(workspaceId,jobId){
    return this.request(`/api/segmentation-workspaces/${encodeURIComponent(workspaceId)}/reconstructions/${encodeURIComponent(jobId)}/review-scene`);
  }
  createReviewScene({workspaceId,jobId,expectedJobVersion,acknowledgeReviewRequired=false}){
    return this.request(`/api/segmentation-workspaces/${encodeURIComponent(workspaceId)}/reconstructions/${encodeURIComponent(jobId)}/review-scene`,{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({expected_job_version:expectedJobVersion,acknowledge_review_required:Boolean(acknowledgeReviewRequired)})
    });
  }
}
