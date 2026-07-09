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
  saveManualMask(workspace,object,session,blob){
    const form=new FormData();
    form.append("edited_mask",blob,"edited-mask.png");
    form.append("base_prompt_revision",String(session.basePromptRevision));
    form.append("base_candidate_index",String(session.baseCandidateIndex));
    form.append("expected_workspace_revision",String(workspace.workspace_revision));
    form.append("expected_object_version",String(object.object_version));
    form.append("expected_manual_revision",String(session.manualRevision));
    return this.request(`/api/segmentation-workspaces/${encodeURIComponent(workspace.workspace_id)}/objects/${object.object_id}/manual-mask`,{method:"PUT",body:form});
  }
  clearManualMask(workspace,object){
    const manual=object.manual_mask||{};
    const query=new URLSearchParams({expected_workspace_revision:workspace.workspace_revision,expected_object_version:object.object_version,expected_manual_revision:manual.manual_revision||0});
    return this.request(`/api/segmentation-workspaces/${encodeURIComponent(workspace.workspace_id)}/objects/${object.object_id}/manual-mask?${query}`,{method:"DELETE"});
  }
}
