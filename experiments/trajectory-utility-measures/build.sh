#!/bin/bash
set -euo pipefail
repo=$1
commit=$2
test "$SLURM_JOB_PARTITION" = lem-cpu-normal
test "$(/usr/bin/git -C "$repo" rev-parse HEAD)" = "$commit"
test -z "$(/usr/bin/git -C "$repo" status --porcelain)"
cd "$repo"
"$HOME/.cargo/bin/rustc" --version
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=64 "$HOME/.cargo/bin/cargo" build \
    --locked --release --jobs 1 --target x86_64-unknown-linux-musl --bin acs2-measure
binary=target/x86_64-unknown-linux-musl/release/acs2-measure
file "$binary"
sha256sum "$binary"
