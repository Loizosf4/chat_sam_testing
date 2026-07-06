# One-Batch Primitive Reconstruction Validation

- Input image: `c:\Users\Loizos\Downloads\images.jpg`
- Output directory: `I:\Code\Chat SAM Testing\chat_sam_testing\clean_room_primitive_reconstruction_20260703_114027`
- Blend file: `I:\Code\Chat SAM Testing\chat_sam_testing\clean_room_primitive_reconstruction_20260703_114027\primitive_reconstruction.blend`
- Semantic object count: 21
- Room/environment object count: 3
- Camera object count: 4
- One-object-one-primitive rule satisfied: True
- Collision count: 6
- Acceptable coarse primitive scene: True
- Later automatic correction pass justified: True

## Semantic Objects
1. `OBJ_001_left_bulletin_board` - left wall bulletin board - plane-like thin rectangular box - loc [-2.92, -1.0, 1.55] - dims [0.07, 0.82, 0.62] - rot [0.0, 0.0, 0.0] - support `left wall` - collisions [] - one primitive True - uncertainty: Notes/details on board intentionally not modeled.
2. `OBJ_002_left_low_filing_cabinet` - left low filing cabinet - cube / rectangular box - loc [-2.45, -1.55, 0.38] - dims [0.58, 0.54, 0.76] - rot [0.0, 0.0, 0.0] - support `floor platform` - collisions [] - one primitive True - uncertainty: Drawer fronts intentionally not modeled.
3. `OBJ_003_left_red_vertical_cabinet` - left red vertical cabinet or side panel - cube / rectangular box - loc [-2.88, -1.55, 0.55] - dims [0.16, 0.32, 1.1] - rot [0.0, 0.0, 0.0] - support `floor platform` - collisions [] - one primitive True - uncertainty: Ambiguous whether cabinet or freestanding panel; represented as one box.
4. `OBJ_004_tall_server_rack` - tall server rack cabinet - cube / rectangular box - loc [-1.72, 0.7, 0.98] - dims [0.78, 0.56, 1.96] - rot [0.0, 0.0, 0.0] - support `floor platform` - collisions [] - one primitive True - uncertainty: Horizontal slats are visible but not modeled because the rack is one object.
5. `OBJ_005_narrow_wall_panel` - narrow wall panel near corner - plane-like thin rectangular box - loc [-0.82, 2.42, 1.65] - dims [0.48, 0.07, 0.72] - rot [0.0, 0.0, 0.0] - support `back wall` - collisions [] - one primitive True - uncertainty: Could be wall art or small cabinet face.
6. `OBJ_006_tall_blue_locker` - tall blue locker cabinet - cube / rectangular box - loc [-0.18, 1.72, 0.94] - dims [0.48, 0.48, 1.88] - rot [0.0, 0.0, 0.0] - support `floor platform` - collisions [] - one primitive True - uncertainty: No handle or panel details modeled.
7. `OBJ_007_center_blue_filing_cabinet` - center blue filing cabinet - cube / rectangular box - loc [-0.68, 0.78, 0.55] - dims [0.58, 0.52, 1.1] - rot [0.0, 0.0, 0.0] - support `floor platform` - collisions [] - one primitive True - uncertainty: Drawer fronts intentionally not modeled.
8. `OBJ_008_desk_table` - desk table - cube / rectangular box - loc [0.28, -0.42, 0.72] - dims [1.78, 1.02, 0.16] - rot [0.0, 0.0, 0.0] - support `floor platform` - collisions ['OBJ_009_chair_behind_desk'] - one primitive True - uncertainty: Table legs intentionally omitted; one tabletop-volume primitive approximates the object.
9. `OBJ_009_chair_behind_desk` - chair behind desk - cube / rectangular box - loc [0.38, 0.22, 0.49] - dims [0.56, 0.48, 0.98] - rot [0.0, 0.0, 0.0] - support `floor platform` - collisions ['OBJ_008_desk_table'] - one primitive True - uncertainty: Chair back/seat/legs not split into separate parts.
10. `OBJ_010_desktop_box_on_table` - small desktop box or monitor on table - cube / rectangular box - loc [0.44, -0.44, 1.035] - dims [0.42, 0.34, 0.47] - rot [0.0, 0.0, 0.0] - support `desk table` - collisions [] - one primitive True - uncertainty: Ambiguous small computer/monitor block on tabletop.
11. `OBJ_011_back_wall_window_blinds` - back wall window with blinds - plane-like thin rectangular box - loc [1.35, 2.42, 1.55] - dims [2.1, 0.07, 0.74] - rot [0.0, 0.0, 0.0] - support `back wall` - collisions ['OBJ_013_right_framed_wall_panel'] - one primitive True - uncertainty: Slats not modeled separately.
12. `OBJ_012_wall_light_bar` - horizontal wall light bar - cylinder - loc [1.28, 2.38, 2.25] - dims [0.09, 0.09, 1.25] - rot [0.0, 90.0, 0.0] - support `back wall` - collisions [] - one primitive True - uncertainty: Single cylinder approximates the glowing light fixture.
13. `OBJ_013_right_framed_wall_panel` - right framed wall panel - plane-like thin rectangular box - loc [2.62, 2.42, 1.55] - dims [0.5, 0.07, 0.72] - rot [0.0, 0.0, 0.0] - support `back wall` - collisions ['OBJ_011_back_wall_window_blinds'] - one primitive True - uncertainty: Frame/detail not modeled.
14. `OBJ_014_right_tall_blue_cabinet` - right tall blue cabinet - cube / rectangular box - loc [1.82, 0.55, 0.65] - dims [0.58, 0.58, 1.3] - rot [0.0, 0.0, 0.0] - support `floor platform` - collisions ['OBJ_015_right_low_blue_cabinet'] - one primitive True - uncertainty: May be part of a storage cluster; kept as distinct visible cabinet block.
15. `OBJ_015_right_low_blue_cabinet` - right low blue cabinet - cube / rectangular box - loc [2.28, 0.05, 0.42] - dims [0.7, 0.62, 0.84] - rot [0.0, 0.0, 0.0] - support `floor platform` - collisions ['OBJ_014_right_tall_blue_cabinet'] - one primitive True - uncertainty: May visually resemble an armchair; represented as one visible cabinet-like block.
16. `OBJ_016_coat_rack_stand` - coat rack stand - cylinder - loc [2.72, -0.22, 0.84] - dims [0.11, 0.11, 1.68] - rot [0.0, 0.0, 0.0] - support `floor platform` - collisions ['OBJ_017_round_hanging_item'] - one primitive True - uncertainty: Hooks and tripod feet intentionally not modeled.
17. `OBJ_017_round_hanging_item` - round hanging item on rack - sphere - loc [2.53, -0.12, 1.24] - dims [0.44, 0.44, 0.44] - rot [0.0, 0.0, 0.0] - support `coat rack stand` - collisions ['OBJ_016_coat_rack_stand'] - one primitive True - uncertainty: Ambiguous hat, lamp shade, or ball-like item; one sphere.
18. `OBJ_018_front_left_floor_box` - front left floor box - cube / rectangular box - loc [-0.82, -1.75, 0.26] - dims [0.58, 0.48, 0.52] - rot [0.0, 0.0, 0.0] - support `floor platform` - collisions [] - one primitive True - uncertainty: Top flaps/details not modeled.
19. `OBJ_019_front_center_floor_box` - front center floor box - cube / rectangular box - loc [0.05, -2.05, 0.3] - dims [0.58, 0.54, 0.6] - rot [0.0, 0.0, 0.0] - support `floor platform` - collisions ['OBJ_020_front_right_floor_box'] - one primitive True - uncertainty: One of the visible foreground box blocks.
20. `OBJ_020_front_right_floor_box` - front right floor box - cube / rectangular box - loc [0.58, -1.84, 0.28] - dims [0.58, 0.48, 0.56] - rot [0.0, 0.0, 0.0] - support `floor platform` - collisions ['OBJ_019_front_center_floor_box', 'OBJ_021_small_floor_box_or_parcel'] - one primitive True - uncertainty: One of the visible foreground box blocks.
21. `OBJ_021_small_floor_box_or_parcel` - small dark floor parcel - cube / rectangular box - loc [0.25, -1.48, 0.42] - dims [0.36, 0.32, 0.28] - rot [0.0, 0.0, 0.0] - support `floor platform` - collisions ['OBJ_020_front_right_floor_box'] - one primitive True - uncertainty: Small ambiguous dark block in foreground cluster.

## Clean-Room Compliance
- input_image_path: c:\Users\Loizos\Downloads\images.jpg
- input_image_exists_at_preflight: True
- only_input_image_used_as_visual_evidence: True
- existing_blend_file_read: False
- existing_manifest_used: False
- prior_transforms_used: False
- prior_reconstruction_outputs_used: False
- external_vision_models_called: False
- blender_operations_used_blender_mcp: True
- shell_based_blender_automation_used: False
- ui_automation_used: False
- post_render_geometry_changes: False
- one_batch_construction: True
- one_object_one_primitive_rule_satisfied: True
- unique_names: True
- semantic_labels_preserved: True
- compliance_result: pass

## Collision Pairs
- OBJ_008_desk_table <-> OBJ_009_chair_behind_desk overlap {'x': 0.56, 'y': 0.11, 'z': 0.16}
- OBJ_011_back_wall_window_blinds <-> OBJ_013_right_framed_wall_panel overlap {'x': 0.03, 'y': 0.07, 'z': 0.72}
- OBJ_014_right_tall_blue_cabinet <-> OBJ_015_right_low_blue_cabinet overlap {'x': 0.18, 'y': 0.1, 'z': 0.84}
- OBJ_016_coat_rack_stand <-> OBJ_017_round_hanging_item overlap {'x': 0.085, 'y': 0.11, 'z': 0.44}
- OBJ_019_front_center_floor_box <-> OBJ_020_front_right_floor_box overlap {'x': 0.05, 'y': 0.3, 'z': 0.56}
- OBJ_020_front_right_floor_box <-> OBJ_021_small_floor_box_or_parcel overlap {'x': 0.14, 'y': 0.04, 'z': 0.28}
