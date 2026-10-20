#!/bin/sh
# eval.sh <tag> [replay options...]: replay every trace raw and robust, stress both, one line each.
# Control: for the stages models, the raw replay must be byte-identical to the old ELF's rig.
W=$HOME/Desktop/stk-work
B=$W/build-robust/bin
tag=$1; shift
out=$W/robust/eval/$tag
mkdir -p $out
for t in $W/robust/trace/*.glb.stkt; do
  m=$(basename $t .glb.stkt)
  g=$W/robust/trace/$m.glb
  [ -f $out/$m.raw.glb ] || $B/skintokens-transfer-replay $t $g $out/$m.raw.glb --raw > /dev/null || { echo "$m raw replay failed"; continue; }
  ctl=""
  [ -f $W/stages/rigged/$m.glb ] && { cmp -s $out/$m.raw.glb $W/stages/rigged/$m.glb && ctl="raw==oldELF" || ctl="RAW-DIFFERS-FROM-OLD-ELF"; }
  info=$($B/skintokens-transfer-replay $t $g $out/$m.robust.glb "$@" | grep -oE 'matched=[0-9]+|inpainted=[0-9]+|fallback=[0-9]+|trimmed_parts=[0-9]+|ms=[0-9]+' | tr '\n' ' ')
  stress() { $B/skintokens-skin-stress $1 | awk '/STRESS invalid/{for(i=1;i<=NF;i++) if($i ~ /^(invalid|torn)=/) printf "%s ", $i} /STRESS spike/{printf "%s %s", $4, $6}'; }
  r=$(stress $out/$m.raw.glb)
  rb=$(stress $out/$m.robust.glb)
  nj=$($B/skintokens-cli glb-info $g | grep -oE '"jointCount":[0-9]+' | cut -d: -f2)
  loss=$(python3 $W/jointloss.py $out/$m.raw.glb $out/$m.robust.glb $nj | sed 's/joints losing >50% of their vertex weight: /lost:/')
  printf "%-20s raw[%s] robust[%s] %s %s %s\n" "$m" "$r" "$rb" "$info" "$loss" "$ctl"
done
