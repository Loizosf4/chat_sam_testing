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
}
