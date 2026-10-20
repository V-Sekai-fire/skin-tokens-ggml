extends SceneTree
func _initialize():
	_run()
func _run():
	var st = load("res://stages/usd_stage.gd").new()
	st.mem_mb = 1024; st.timeout_units = 4000
	root.add_child(st)
	await process_frame
	print("init ", st.init())
	for f in OS.get_cmdline_user_args():
		print("open ", f, ": ", st.open_file("res://" + f).left(300))
		print(" meshes ", st.count(), " skeletons ", st.call_now("usd_skeleton_count"), " anims ", st.call_now("usd_skel_anim_count"))
		var si: Dictionary = st.call_now("usd_skeleton_info", [0])
		print(" skel ", si.get("path"), " njoints ", si.get("njoints"), " animation ", si.get("animation"), " keys ", si.keys())
		print(" mesh0 ", st.call_now("usd_mesh_info", [0]))
		print(" skin0 ", st.call_now("usd_mesh_skin", [0]))
		print(" anim0 ", st.call_now("usd_skel_anim_info", [0]))
	quit()
