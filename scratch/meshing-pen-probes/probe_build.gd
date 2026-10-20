extends SceneTree
# usd.elf + util/usd_skel.gd: build avatar.usdz, play clips, dump bone globals.
const UsdSkel := preload("res://util/usd_skel.gd")
func _initialize():
	_run()
func _run():
	var a := OS.get_cmdline_user_args()
	var st = load("res://stages/usd_stage.gd").new()
	st.mem_mb = 1024; st.timeout_units = 4000
	root.add_child(st)
	await process_frame
	st.init()
	print("open: ", st.open_file("res://" + a[0]).left(120))
	var t0 := Time.get_ticks_msec()
	var scene := UsdSkel.build(st, "Avatar")
	print("build ms ", Time.get_ticks_msec() - t0)
	root.add_child(scene)
	var sk: Skeleton3D = scene.get_node("Skeleton3D")
	var ap: AnimationPlayer = scene.get_node("AnimationPlayer")
	print("bones ", sk.get_bone_count(), " meshes ", sk.get_child_count(), " clips ", ap.get_animation_list().size())
	var bones := FileAccess.get_file_as_string(a[1] + "/bones.txt").split("\n")
	var out := PackedFloat32Array()
	for nm in ["sit_06", "walk_06"]:
		ap.play(nm); ap.pause()
		for f in [0, 77, 150, 301]:
			ap.seek(f / 30.0, true)
			await process_frame
			for bn in bones:
				var g := sk.get_bone_global_pose(sk.find_bone(bn))
				out.append_array([g.origin.x, g.origin.y, g.origin.z])
	var fa := FileAccess.open(a[2], FileAccess.WRITE); fa.store_buffer(out.to_byte_array()); fa.close()
	quit()
