# One-Batch Clean-Room Primitive Reconstruction Validation

- Input image: `i:\Docs\Images\2d office.jpg`
- Blend: `I:\Code\Chat SAM Testing\chat_sam_testing\clean_room_primitive_reconstruction_20260706_104253\primitive_reconstruction.blend`
- Semantic objects: 15
- Room/environment objects: 12
- Cameras: 4
- Collision count: 4
- Clean-room compliance: pass
- One-object-one-primitive rule: satisfied
- Acceptable coarse scene: no
- Later automatic correction justified: yes

## Semantic Objects
1. large executive desk - cube - loc [-1.75, -0.95, 0.38] - dims [2.85, 1.55, 0.76] - support floor
2. long low credenza under windows - cube - loc [-3.05, 1.25, 0.35] - dims [2.75, 0.55, 0.7] - support floor
3. high-back office chair - cube - loc [-3.05, -1.55, 0.78] - dims [0.78, 0.85, 1.55] - support floor
4. left visitor chair behind desk - cube - loc [-0.95, 0.18, 0.57] - dims [0.55, 0.58, 1.08] - support floor
5. center visitor chair behind desk - cube - loc [-0.15, 0.36, 0.56] - dims [0.55, 0.6, 1.05] - support floor
6. rear armchair by window - cube - loc [0.72, 1.35, 0.52] - dims [0.82, 0.78, 1.02] - support floor
7. small wooden coffee table - cube - loc [1.22, 0.85, 0.3] - dims [1.18, 0.68, 0.44] - support floor
8. center white lounge armchair - cube - loc [2.22, 0.56, 0.55] - dims [0.92, 0.9, 1.02] - support floor
9. cropped foreground white sofa - cube - loc [3.15, -2.28, 0.45] - dims [2.1, 1.0, 0.9] - support floor
10. wall-mounted television - cube - loc [1.35, 3.28, 1.88] - dims [1.32, 0.07, 0.76] - support right feature wall
11. oval illuminated wall feature - cylinder - loc [3.08, 3.25, 2.2] - dims [1.82, 0.08, 0.72] - support right feature wall
12. large curved illuminated wall feature - cube - loc [2.55, 3.23, 1.1] - dims [2.75, 0.07, 0.62] - support right feature wall
13. linear pendant light above desk - cube - loc [-1.55, 0.52, 2.42] - dims [1.18, 0.13, 0.08] - support ceiling
14. center recessed ceiling light - cube - loc [0.38, 1.85, 3.15] - dims [0.32, 0.22, 0.05] - support ceiling
15. right recessed ceiling light - cube - loc [3.35, 2.05, 3.15] - dims [0.34, 0.23, 0.05] - support ceiling

## Collision Pairs
- 01_executive_desk <-> 03_high_back_office_chair overlap (0.684, 0.845, 0.755)
- 01_executive_desk <-> 04_left_visitor_chair overlap (0.608, 0.13, 0.73)
- 06_rear_left_armchair <-> 07_wood_coffee_table overlap (0.55, 0.283, 0.44)
- 07_wood_coffee_table <-> 08_center_lounge_armchair overlap (0.213, 0.67, 0.44)

## Uncertainty Notes
- 01_executive_desk: One box approximates the whole desk; legs/drawers omitted.
- 02_window_credenza: Length partly occluded by desk and chair.
- 03_high_back_office_chair: One tall box covers seat/back/base silhouette.
- 04_left_visitor_chair: partly occluded by desk/chairs; scale and depth are approximate
- 05_center_visitor_chair: partly occluded by desk/chairs; scale and depth are approximate
- 06_rear_left_armchair: partly occluded by desk/chairs; scale and depth are approximate
- 07_wood_coffee_table: Visible rectangular block near lounge chairs.
- 08_center_lounge_armchair: Box approximates whole armchair without arms/legs.
- 09_foreground_sofa: cropped by input frame; depth and full footprint are uncertain
- 10_wall_television: Flat black box on back/right wall.
- 11_oval_wall_feature: Single flattened cylinder represents oval fixture; shelves omitted as detail.
- 12_curved_wall_feature: curved source feature approximated by a straight box primitive
- 13_pendant_light: Cable omitted; bar represents entire visible pendant fixture.
- 14_center_recessed_ceiling_light: One flat box for the fixture.
- 15_right_recessed_ceiling_light: One flat box for the fixture.
