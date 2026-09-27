#!/usr/bin/env bash
# Download Spider 1.0 and BIRD into data/raw/ and unpack them under data/. Every archive is checked against a
# pinned SHA-256, so a changed upstream file fails loudly instead of silently changing the benchmark.
# Usage: scripts/download_data.sh [--with-bird-train]   (BIRD train is 8.9 GB, so it is opt-in)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

# Official sources: https://yale-lily.github.io/spider (Google Drive) and https://bird-bench.github.io
SPIDER_URL="https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t"
BIRD_URL=https://bird-bench.oss-cn-beijing.aliyuncs.com

fetch() {  # url, file name, expected sha256 (empty: print it so it can be pinned)
  local url=$1 out=data/raw/$2 want=$3
  if [ ! -f "$out" ]; then
    curl -fL --retry 5 --retry-delay 10 -C - --no-progress-meter -o "$out.part" "$url"
    # Google Drive answers with an HTML page instead of the file when its download flow changes.
    [ "$(head -c 2 "$out.part")" = PK ] || { echo "$url did not return a zip; see $out.part" >&2; exit 1; }
    mv "$out.part" "$out"
  fi
  local got
  got=$(sha256sum "$out" | cut -d' ' -f1)
  if [ -n "$want" ] && [ "$got" != "$want" ]; then
    echo "SHA-256 mismatch for $out: got $got, want $want" >&2
    exit 1
  fi
  echo "$got  $out"
}

unpack() {  # zip, path prefix inside it, destination; skips macOS metadata entries
  python3 - "$@" <<'EOF'
import pathlib, shutil, sys, zipfile
src, prefix, dest = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]).resolve()
with zipfile.ZipFile(src) as z:
    for info in z.infolist():
        name = info.filename
        if info.is_dir() or not name.startswith(prefix) or "__MACOSX" in name or name.endswith(".DS_Store"):
            continue
        target = (dest / name[len(prefix):]).resolve()
        if not target.is_relative_to(dest):
            sys.exit(f"refusing to write outside {dest}: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(info) as fsrc, open(target, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)
EOF
}

fetch "$SPIDER_URL" spider_data.zip 00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b
fetch "$BIRD_URL/dev.zip" bird_dev.zip cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630
fetch "$BIRD_URL/minidev.zip" bird_minidev.zip cc48ba16838204e4e214512030cb572eeb5f7bcdd999bae4b9b6ff12ec13b92f
if [ "${1:-}" = --with-bird-train ]; then
  fetch "$BIRD_URL/train.zip" bird_train.zip ""
fi

# Layout: data/spider/{database/,dev.json,...}, data/bird/dev/{dev.json,dev_databases/,...},
# data/bird/minidev/{mini_dev_sqlite.json,dev_databases/,...}
[ -d data/spider/database ] || unpack data/raw/spider_data.zip spider_data/ data/spider
if [ ! -d data/bird/dev/dev_databases ]; then
  unpack data/raw/bird_dev.zip dev_20240627/ data/bird/dev   # the June 2024 corrected dev release
  unpack data/bird/dev/dev_databases.zip dev_databases/ data/bird/dev/dev_databases
  rm data/bird/dev/dev_databases.zip
fi
[ -d data/bird/minidev/dev_databases ] || unpack data/raw/bird_minidev.zip minidev/MINIDEV/ data/bird/minidev
echo "unpacked: $(find data/spider/database -name '*.sqlite' | wc -l) Spider DBs," \
  "$(find data/bird/dev/dev_databases -name '*.sqlite' | wc -l) BIRD dev DBs," \
  "$(find data/bird/minidev/dev_databases -name '*.sqlite' | wc -l) BIRD mini-dev DBs"
