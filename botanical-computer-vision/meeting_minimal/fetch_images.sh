#!/usr/bin/env bash
# Download the 1,641 strict-set images (gq) and thumbnails (tn) into /home/user/imgcache (never committed).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p /home/user/imgcache/gq /home/user/imgcache/tn
python3 -c "
import csv
for r in csv.DictReader(open('organ_benchmark/split_manifests/master_manifest.csv')):
    fn = r['image_id'].replace('/', '_') + '.jpg'
    print(r['image_url'], '/home/user/imgcache/gq/' + fn); print(r['thumbnail_url'], '/home/user/imgcache/tn/' + fn)
" | xargs -P 24 -n 2 sh -c '[ -s "$1" ] || curl -sS --retry 2 -o "$1" "$0"'
