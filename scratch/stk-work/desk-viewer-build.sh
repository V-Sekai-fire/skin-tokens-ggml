set -e
B=/root/skintokens-viewer-build
mkdir -p $B/engine $B/w && cd $B
tar -xf /mnt/c/Users/ernest.lee/viewer-build-small.tar && rm /mnt/c/Users/ernest.lee/viewer-build-small.tar
cp LICENSE-model-explorer.txt engine/
R=https://github.com/V-Sekai-fire/service-godot-build/releases/download/v20260930-double.1
cd engine
for f in godot.macos.template_release.double.arm64 godot.windows.template_release.double.x86_64.llvm.exe godot.linuxbsd.template_release.double.x86_64 SHA256SUMS; do curl -sSfL -o $f $R/$f; done
sha256sum -c SHA256SUMS --ignore-missing
cd ../w
H=https://huggingface.co/chibifire/SkinTokens-GGUF-BF16/resolve/912d4d1
for f in mesh-encoder.gguf skin-vae.gguf tokenrig.gguf; do curl -sSfL -o $f "$H/$f"; done
ls -la; sha256sum *.gguf
echo VIEWER-BUILD-READY
