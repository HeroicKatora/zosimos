#!/bin/bash
owndir="$(dirname "$(readlink -f "$0")")"
pushd "$owndir"
RUSTFLAGS=--cfg=web_sys_unstable_apis cargo build --no-default-features --target wasm32-unknown-unknown --lib -p stealth-paint-editor
wasm-bindgen --out-dir target/generated --web target/wasm32-unknown-unknown/debug/stealth_paint_editor.wasm
python -m http.server --directory target/generated 8000
