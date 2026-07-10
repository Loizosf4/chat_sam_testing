import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src import compile_unified_v3_scene as cli
from src import unified_v3_scene_compiler as compiler
from src.support_graph import infer_support_graph


def make_inputs(root: Path, masks: list[tuple[str,str,str|None]] | None = None) -> tuple[Path,Path,Path]:
    width,height=32,24;root.mkdir(parents=True,exist_ok=True);sam=root/"sam";moge=root/"moge";sam.mkdir();moge.mkdir()
    source=root/"source.png";Image.new("RGB",(width,height),(80,90,100)).save(source)
    entries=[]
    for index,(mask_id,label,color) in enumerate(masks or [("stable-1","arbitrary / label?!",None)]):
        filename=f"mask-{index}.png";array=np.zeros((height,width),np.uint8);array[4+index:10+index,5+index:13+index]=255;Image.fromarray(array).save(sam/filename)
        item={"mask_id":mask_id,"label":label,"filename":filename}
        if color is not None:item["color"]=color
        entries.append(item)
    (sam/"metadata.json").write_text(json.dumps({"image_id":"image","width":width,"height":height,"masks":entries}),encoding="utf-8")
    yy,xx=np.mgrid[:height,:width];depth=np.ones((height,width),np.float32);points=np.dstack(((xx-width/2)/20,(yy-height/2)/20,depth)).astype(np.float32);normal=np.zeros_like(points);normal[...,1]=-1
    np.savez(moge/"geometry.npz",points=points,depth=depth,valid_mask=np.ones((height,width),bool),intrinsics=np.eye(3),normal=normal)
    (moge/"metadata.json").write_text(json.dumps({"source_image_dimensions":{"width":width,"height":height},"estimated_fov_x_degrees":60.0}),encoding="utf-8")
    return sam,source,moge


def test_input_validation_preserves_ids_duplicate_labels_and_fallback_colors(tmp_path):
    sam,source,moge=make_inputs(tmp_path,[("stable-a","same",None),("stable-b","same","bad")])
    result=compiler.validate_compiler_inputs(sam_dir=sam,source_path=source,moge_dir=moge)
    assert [item["mask_id"] for item in result["metadata"]["masks"]]==["stable-a","stable-b"]
    assert [item["label"] for item in result["metadata"]["masks"]]==["same","same"]
    assert result["fallback_color_object_ids"]==["stable-a","stable-b"]
    assert all(item["color"].startswith("#") for item in result["metadata"]["masks"])


@pytest.mark.parametrize(
    ("mutation","message"),
    [
        (lambda sam,source,moge:(sam/"metadata.json").unlink(),"SAM metadata is missing"),
        (lambda sam,source,moge:(sam/"metadata.json").write_text(json.dumps({"image_id":"x","width":32,"height":24,"masks":[]}),encoding="utf-8"),"at least one mask"),
        (lambda sam,source,moge:_edit_metadata(sam,lambda data:data["masks"].append(dict(data["masks"][0]))),"duplicate SAM mask_id"),
        (lambda sam,source,moge:_edit_metadata(sam,lambda data:data["masks"][0].update(filename="../mask.png")),"unsafe filename"),
        (lambda sam,source,moge:(sam/"mask-0.png").unlink(),"mask file is missing"),
        (lambda sam,source,moge:Image.fromarray(np.full((24,32),12,np.uint8)).save(sam/"mask-0.png"),"not binary"),
        (lambda sam,source,moge:Image.fromarray(np.zeros((24,32),np.uint8)).save(sam/"mask-0.png"),"is empty"),
        (lambda sam,source,moge:Image.fromarray(np.ones((4,4),np.uint8)*255).save(sam/"mask-0.png"),"dimensions"),
        (lambda sam,source,moge:Image.new("RGB",(4,4)).save(source),"source image dimensions"),
        (lambda sam,source,moge:_remove_moge_array(moge,"normal"),"missing arrays"),
    ],
)
def test_input_validation_failures_are_clear(tmp_path,mutation,message):
    sam,source,moge=make_inputs(tmp_path);mutation(sam,source,moge)
    with pytest.raises(compiler.CompilerValidationError,match=message):
        compiler.validate_compiler_inputs(sam_dir=sam,source_path=source,moge_dir=moge)


def _edit_metadata(sam:Path,edit)->None:
    path=sam/"metadata.json";data=json.loads(path.read_text());edit(data);path.write_text(json.dumps(data),encoding="utf-8")


def _remove_moge_array(moge:Path,name:str)->None:
    with np.load(moge/"geometry.npz",allow_pickle=False) as archive:data={key:archive[key] for key in archive.files if key!=name}
    np.savez(moge/"geometry.npz",**data)


def test_one_object_support_is_explicitly_unknown_without_min_or_max_failure():
    mask=np.zeros((20,20),bool);mask[5:10,5:10]=True
    points=np.column_stack([np.linspace(0,.1,25),np.linspace(0,.1,25),np.full(25,.8)])
    item={"object_id":"only","semantic_label":"unrecognized","points":points,"normals":np.tile([0,0,1.],(25,1)),"mask":mask,"depth":np.ones(25)}
    result=infer_support_graph([item])
    assert result["assignments"]["only"]=={"target":None,"type":"unknown","confidence":.25}


def test_generic_quality_gates_use_ids_not_labels(tmp_path):
    guard=compiler.CleanReadGuard([]);objects=[_quality_object("a","duplicate"),_quality_object("b","duplicate")]
    scene={"semantic_objects":objects,"room_proxies":[{"plane_id":"plane_floor"}],"camera_candidates":[{"camera_id":"camera"}],"provisional_camera_id":"camera","support_graph":{"final_contact_propagation":{"maximum_final_gap":0.0}}}
    gates,violations=compiler.evaluate_generic_quality_gates(scene=scene,export_ids=["a","b"],collisions=[],room_fit={"passed":True},guard=guard)
    assert violations==[] and all(gates.values())
    assert "exactly_20_semantic_cubes" not in gates
    assert not any("desk" in key or "chair" in key for key in gates)


def _quality_object(object_id:str,label:str)->dict:
    return {"object_id":object_id,"semantic_label":label,"primitive_type":"cube","support_target":None,"orientation_method":"normal_first_v3_universal","placement_classification":"placement_high_confidence","transform":{"center":[0,0,1],"dimensions":[1,1,1],"rotation_matrix":np.eye(3).tolist(),"quaternion_wxyz":[1,0,0,0]}}


def test_handoff_is_count_aware_and_copies_exact_manifest(tmp_path):
    output=tmp_path/"output";handoff=tmp_path/"handoff";output.mkdir()
    manifest={"room_proxies":[{"plane_id":"floor"}],"semantic_primitive_count":2,"semantic_primitives":[{"object_id":"a"},{"object_id":"b"}]}
    (output/"blender_one_batch_manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    (output/"allowed_inputs_manifest.json").write_text("{}",encoding="utf-8")
    compiler.finalize_handoff(output_dir=output,handoff_dir=handoff,semantic_object_count=2)
    assert json.loads((handoff/"blender_one_batch_manifest.json").read_text())==manifest
    assert json.loads((handoff/"expected_output_contract.json").read_text())["semantic_primitive_count"]==2
    text=(handoff/"README.md").read_text()+(handoff/"execution_instructions.md").read_text()
    assert "2 semantic primitives" in text and "20" not in text and "desktop_box" not in text


def test_structural_scope_failure_is_atomic_and_removes_staging(tmp_path):
    sam,source,moge=make_inputs(tmp_path/"inputs");output=tmp_path/"result"
    with pytest.raises(compiler.CompilerValidationError,match="indoor room reconstruction requires"):
        compiler.compile_clean(sam_dir=sam,source_path=source,moge_dir=moge,output_dir=output,scene_id="synthetic",handoff_dir=None)
    assert not output.exists()
    assert not list(tmp_path.glob(".result.staging-*"))


def test_existing_output_is_not_overwritten(tmp_path):
    sam,source,moge=make_inputs(tmp_path/"inputs");output=tmp_path/"result";output.mkdir();marker=output/"keep";marker.write_text("unchanged")
    with pytest.raises(FileExistsError,match="refusing to overwrite"):
        compiler.compile_clean(sam_dir=sam,source_path=source,moge_dir=moge,output_dir=output,handoff_dir=None)
    assert marker.read_text()=="unchanged"


def test_custom_compile_publishes_without_touching_global_handoff(tmp_path,monkeypatch):
    sam,source,moge=make_inputs(tmp_path/"inputs");output=tmp_path/"result";global_handoff=tmp_path/"global";global_handoff.mkdir();marker=global_handoff/"keep";marker.write_text("unchanged")
    monkeypatch.setattr(compiler,"HANDOFF",global_handoff)
    def fake_compile(**kwargs):
        (kwargs["output"]/"marker.json").write_text("{}")
        return {"scene_id":"custom","semantic_object_count":1}
    monkeypatch.setattr(compiler,"_compile_clean_into",fake_compile)
    monkeypatch.setattr(compiler,"_validate_compiled_output",lambda output_dir,object_ids:None)
    result=compiler.compile_clean(sam_dir=sam,source_path=source,moge_dir=moge,output_dir=output,scene_id="custom")
    assert result["semantic_object_count"]==1 and (output/"marker.json").is_file()
    assert marker.read_text()=="unchanged" and list(global_handoff.iterdir())==[marker]


def test_validation_failure_after_staging_leaves_no_output(tmp_path,monkeypatch):
    sam,source,moge=make_inputs(tmp_path/"inputs");output=tmp_path/"result"
    def fake_compile(**kwargs):
        (kwargs["output"]/"partial").write_text("partial")
        return {"scene_id":"custom","semantic_object_count":1}
    monkeypatch.setattr(compiler,"_compile_clean_into",fake_compile)
    monkeypatch.setattr(compiler,"_validate_compiled_output",lambda output_dir,object_ids:(_ for _ in ()).throw(RuntimeError("contract invalid")))
    with pytest.raises(RuntimeError,match="contract invalid"):
        compiler.compile_clean(sam_dir=sam,source_path=source,moge_dir=moge,output_dir=output,handoff_dir=None)
    assert not output.exists() and not list(tmp_path.glob(".result.staging-*"))


def test_cli_invalid_input_is_nonzero_without_traceback(tmp_path,capsys):
    code=cli.main(["--mode","clean_reconstruction","--sam-dir",str(tmp_path/"missing"),"--source-image",str(tmp_path/"missing.png"),"--moge-dir",str(tmp_path/"missing-moge"),"--output-dir",str(tmp_path/"out"),"--scene-id","scene"])
    captured=capsys.readouterr()
    assert code==2 and "Traceback" not in captured.err and captured.out==""


def test_cli_summary_is_machine_readable(tmp_path,monkeypatch,capsys):
    output=tmp_path/"out"
    def fake_compile(**kwargs):
        output.mkdir();(output/"compilation_report.json").write_text(json.dumps({"passed":False}))
        return {"scene_id":"generic","semantic_object_count":2}
    monkeypatch.setattr(cli,"compile_clean",fake_compile)
    code=cli.main(["--output-dir",str(output),"--scene-id","generic"])
    summary=json.loads(capsys.readouterr().out)
    assert code==0 and summary["success"] is True and summary["object_count"]==2 and summary["compilation_passed"] is False


def test_regression_audit_mode_remains_callable(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(cli,"run_regression_audit",lambda clean,output:{"counts":{"agreement":1}})
    assert cli.main(["--mode","regression_audit","--output-dir",str(tmp_path/"audit")])==0
    assert json.loads(capsys.readouterr().out)["mode"]=="regression_audit"
