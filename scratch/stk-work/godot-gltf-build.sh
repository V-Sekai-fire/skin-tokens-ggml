cd ~/Desktop/stk-work/wt/godot-gltf
scons platform=macos arch=arm64 target=editor precision=double -j10 > ~/Desktop/stk-work/godot-gltf-build.log 2>&1
echo "BUILD-EXIT $?" >> ~/Desktop/stk-work/godot-gltf-build.log
ls -la bin/ >> ~/Desktop/stk-work/godot-gltf-build.log
