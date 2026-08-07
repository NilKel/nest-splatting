# Quest Local Viewer — Plan

**This file:** `/home/nilkel/Projects/nest-splatting/docs/QUEST_LOCAL_VIEWER_PLAN.md`
**Workstation SSH:** `nilkel@10.176.128.124` (LAN — SSH here from Mac to read anything referenced below)

---

## Goal

Build a viewer that runs **locally on the Mac** and renders our baked scenes on the Mac GPU. Streaming to the Meta Quest 2 comes later, but we already have significant in-progress code for that too (see "Streaming" section — the Mac→Quest UDP+HEVC pipeline in `Halloumi-Stream` is the answer, not the "get a Windows PC" workaround I was pushing earlier).

## Where things stand across the Quest work

There are **six related repos on this workstation**. All at `/home/nilkel/Projects/`.

### 1. `Halloumi-Quest` — native Vulkan + OpenXR APK, first attempt
`/home/nilkel/Projects/Halloumi-Quest/` (last touched 2026-07-10)

- `app/src/main/cpp/` + `app/src/main/java/` — the Vulkan + OpenXR renderer, Android glue
- `docs/PLAN.md`, `docs/ARCHITECTURE.md` — port plan and pipeline design
- **Status per its README:** "in development"
- **Why native instead of WebXR:** *"Meta Quest Browser's WebGPU implementation on Adreno 650 crashes the GPU context during execution — for any combination of Halloumi's compute passes, and even for a minimal cleared-only render pass. A minimal WebGPU triangle works, but Halloumi's resource graph does not. That's the browser/driver floor as it stands today on Quest 2 hardware."* This is a much sharper statement than "Adreno is slow" — it's actively broken for our compute graph.

### 2. `Halloumi-Quest-Fork` — the iteration branch of #1 (source for the shipping APKs)
`/home/nilkel/Projects/Halloumi-Quest-Fork/` (last touched 2026-07-10)

- Same architecture as Halloumi-Quest, forked to iterate on
- `docs/V0_3_20_MULTIVIEW_PLAN.md` — the multiview (single-pass stereo) plan
- Compiled outputs land in `/home/nilkel/Projects/bitymi-demos/xr-native/halloumi-quest-fork-vX.Y.Z-dev.apk` — you'll see **v0.3.7 through v0.3.38c** sitting there, so this fork has had a lot of tuning passes.
- Sideload URL: `https://nilkel.github.io/bitymi-demos/xr-native/`

### 3. `Halloumi-Stream` — Mac renders, Quest displays (the streaming path)
`/home/nilkel/Projects/Halloumi-Stream/` (last touched **2026-07-23 — most recent**)

**This is exactly the thing you're asking about.** Its README, verbatim:
> Mac (Apple Silicon) renders 2DGS + SV splats with the full nest-splatting lean-CONIC pipeline; Quest 2 displays them as a thin HEVC video client. No Meta first-party Mac software exists, so this is a custom UDP-over-Wi-Fi pipeline both ends of which live in this repo.

Layout:
- `mac/` — Rust binary. Contains `Cargo.toml` + `src/`. Renders on the Mac's GPU via wgpu, encodes via VideoToolbox HEVC, sends UDP.
- `quest/` — C++ skeletons + a `PATCHES.md` that specifies how to graft them into `Halloumi-Quest-Fork`. Files:
  - `udp_frame_recv.cpp` — receive HEVC frames from Mac
  - `mediacodec_dec.cpp` — hardware HEVC decode (Adreno 650's video pipeline, not its compute — this is fine)
  - `udp_pose_send.cpp` — send OpenXR head pose back to Mac
  - `stream_client.h` — API surface
- `proto/messages.rs` — shared UDP wire format. If you change one side, change both.
- `docs/DESIGN.md` — full architecture, M1–M5 milestones, latency budget, build/setup
- `docs/PROTOCOL.md` — UDP wire format spec

**Status per its README** (2026-07-23):
- **M1 scaffold**: done. Rust binary + wgpu test-pattern renderer + VideoToolbox HEVC encoder + UDP frame sender all wired. Compiles; needs a Mac to actually run VideoToolbox and needs a Quest APK to receive.
- **M2 scaffold**: done. Pose receiver on Mac done; pose sender skeleton on Quest done. Not yet integrated into the OpenXR loop on the Quest side.
- **Quest side**: `quest/` has the C++ skeletons + PATCHES.md; nothing built yet on the APK side — needs a `stream-client-mvp` branch cut on Halloumi-Quest-Fork + CMakeLists update.
- **M3+ (real splat renderer, stereo, polish)**: designed on paper; code stubs in `mac/src/render.rs` with TODO markers.

This is your straight line — finish M3 on the `mac/` side (swap the test pattern for the real Halloumi-WS pipeline ported to wgpu-rs), cut the `stream-client-mvp` branch on Halloumi-Quest-Fork, drop the `quest/` files in per PATCHES.md, ship a new APK, and you have Mac→Quest streaming without any Windows machine anywhere in the loop.

### 4. `Halloumi-XR` — WebXR + WebGPU fork of Halloumi-WS
`/home/nilkel/Projects/Halloumi-XR/`

- Vite/TypeScript project (source layout matches Halloumi-WS)
- Head tracking passes WebXR view matrices straight into the camera uniform; controllers do orbit-style navigation via thumbsticks (`left X/Y → pan`, `right Y → dolly`, `right X → yaw`)
- Targets `XRGPUBinding` (WebGPU inside an `immersive-vr` session)
- **Status per its README:** "first-pass on-device tuning required" — the code compiles and runs, but it's the target of the Adreno WebGPU driver crash described under Halloumi-Quest above. Superseded by the native path for on-device, but still the reference implementation for what a stereo-per-eye render loop of our pipeline should look like.
- Deployed at `bitymi-demos/xr/` and `xr-c/` cards.

### 5. `Quest-OpenXR-Vulkan` — third-party minimal OpenXR+Vulkan Android reference
`/home/nilkel/Projects/Quest-OpenXR-Vulkan/`

- Not ours. A minimal single-directory Quest OpenXR+Vulkan sample.
- Kept around as a "how little is actually needed to run a native app" reference — the Android/Gradle scaffolding is much slimmer than the Meta samples.

### 6. `Quest-XR` — third-party OpenXR SDK sample fork
`/home/nilkel/Projects/Quest-XR/`

- Not ours. The stock Khronos `OpenXR-SDK-Source` sample, packaged as a Gradle/CMake Android project.
- Probably the seed that `Halloumi-Quest-Fork` was cut from — same "Quest XR" boilerplate is visible in the fork's README.

## Where the baked CONIC renderer lives

Two implementations. Bit-identical output; use whichever fits the target.

### CUDA reference (training / benchmarks / bake-time renders)
`/home/nilkel/Projects/nest-splatting/submodules/diff_surfel_bake_render_lean/`

- `cuda_rasterizer/forward.cu` — CONIC-corrected forward: linearized ray-splat, fp16 collab-mem, atlas UV precompute, single BC7/typeD sample per fragment. Bit-identical to prod (`diff_surfel_bake_render/`) but ~86% faster on a 5090 for the mip360 mean.
- `cuda_rasterizer/rasterizer_impl.cu` — tile binning, sort, dispatch
- `__init__.py` — Python wrapper
- `setup.py` — build glue (`conda run -n nest_splatting python -m pip install -e . --no-build-isolation`)

Siblings you may want later:
- `submodules/diff_surfel_bake_render_lean_occ/` — adds proxy-mesh Z-cull (mesh_cull=1)
- `submodules/diff_surfel_bake_render_paired_lean/` — for `res_3d_paired` bakes (not needed for the brain)

### WebGPU / WGSL implementation (Halloumi-WS shipping today)
`/home/nilkel/Projects/Halloumi-WS/src/shaders/`

- `render_2dgs.wgsl` — fragment shader mirroring the CUDA CONIC path. Atlas UV precompute + single HW BC7/ASTC fetch per fragment.
- `surfel_cull.wgsl` — per-Gauss cull + compact into `Splat2DGS` (96 B / Gauss, layout in header comment)
- `preprocess_2dgs.wgsl` — per-splat transmat + screen-space projection + atlas UV precompute (fills `Splat2DGS.uv_base_*`, `uv_scale_*`, `layer`)
- `radix_*.wgsl` — wait-free hierarchical Blelloch sort (24-bit key + 8-bit LSB tiebreak)
- `tile_raster_2dgs.wgsl` — tile-based dispatch to fragment
- `display.wgsl` — final blit

Host / TS glue:
- `src/splat-app.ts` — main loop, camera + control, URL params, postMessage sync bridge (added 2026-07-28)
- `src/gaussian-renderer.ts` — WebGPU pipeline: bind groups, buffer allocation, per-frame dispatch order
- `src/ply-loader.ts` + `src/utils/loaders/ply/Nat2Parser.ts` — `.bitymi` chunk container + `.nat2` atlas parser (typeD BC7 = atlas_format=7, typeD ASTC = atlas_format=8)
- `src/camera.ts` + `src/camera-control.ts` — orbit camera + impulse-based drag/pan/wheel

## Brain checkpoint — what to copy to the Mac

The 3D_SH_res N-variant bake shipping as `brain_2k.bitymi`:

**Bake output directory (2.5 GB, only needed if re-baking):**
`/home/nilkel/Projects/nest-splatting/outputs/brain/clip_imgVol_47um_worldLight_wet_rev3/3D_SH_res/Nhimn2_7_12_19_SV_30thr_005w50gLP4l_C2frbg_3kbs_r2/baked_atlas/`

Contents: `baked.ply` / `baked.bply`, `atlas_texture.bc7`, `atlas_texture.pt`, `atlas_rects.pt`, `bake_meta.json`, three `scene_astc*.nat2` variants, `renders/`, `benchmark_results.json`.

**Packaged bundle (single 75 MB file — copy this):**
`/home/nilkel/Projects/bitymi-demos/scenes/himalaya_brain/brain_2k.bitymi`

ASTC variant (same scene, typeD ASTC — worth testing Apple Silicon's native ASTC decode):
`/home/nilkel/Projects/bitymi-demos/scenes/himalaya_brain/brain_2k_astc.bitymi`

`.bitymi` layout: 8-byte magic `BITYMI01`, u32 chunk count, per-chunk `[u64 offset, u64 size, u32 name_len, name]` headers, chunk data. Chunks: `scene.bply` + `scene.nat2`. Parser is `Halloumi-WS/src/ply-loader.ts` + `Nat2Parser.ts`.

## Deploy / re-bake pipeline

Only needed when producing a new `.bitymi` from a fresh bake.

- `scripts/export_textures_bin.py` — bake output dir → `.nat2` (typeD BC7 with `--bc7-codebook`, typeD ASTC with `--astc-codebook`)
- `scripts/build_bc7_bundles_fp16.py` — nat2 → `.bitymi` chunk container
- `docs/BITYMI_BUNDLES.md`, `docs/DEPLOY_DEMO.md`, `docs/BAKED_RENDERING.md` — full pipeline reference

## Streaming path — updated with Halloumi-Stream

Options ranked by what actually gets the user's Mac talking to the Quest fastest.

### 0. Finish `Halloumi-Stream` — Mac→Quest UDP+HEVC (recommended)
No Windows machine required. Already scaffolded on both sides at 2026-07-23. Remaining work per its own README:

- **Mac side (`mac/`)**: swap the wgpu test pattern for the real Halloumi-WS pipeline ported to wgpu-rs. The WGSL shaders in `Halloumi-WS/src/shaders/` port 1:1 (wgpu-rs uses the same shader language + same buffer semantics). Approx 800 LOC of TS host code needs Rust equivalents.
- **Quest side (`quest/` → `Halloumi-Quest-Fork`)**: cut a `stream-client-mvp` branch on the fork, drop the four `quest/*.cpp/.h` files in per `PATCHES.md`, update CMakeLists, build APK.
- **M3+ polish**: stereo, real head-pose integration into OpenXR loop, latency budget tuning (target < 40 ms motion-to-photon).

Estimate: 1–2 weeks if the wgpu-rs port of the compute pipeline goes cleanly.

### 1. Windows PC + Air Link (fallback if Halloumi-Stream stalls)
Existing `bitymi-demos/xr/` viewer + Windows machine + Oculus PC app. Zero code changes; hardware buy-in ~$300 mini-PC.

### 2. Continue pushing native on-device (`Halloumi-Quest-Fork`)
The v0.3.7→v0.3.38c series has been iterating on this. Not blocked in principle — Vulkan on Adreno 650 works fine, it's WebGPU that crashes. The question is whether the Quest 2 hardware can hit target frame times for our scene sizes at all, or whether we're brute-forcing a mobile GPU that fundamentally can't. The v0.3.20+ multiview plan (`Halloumi-Quest-Fork/docs/V0_3_20_MULTIVIEW_PLAN.md`) is the next lever — single-pass stereo halves the geometry work.

### 3. WebGPU + WebXR on macOS
Still not real. Apple hasn't adopted OpenXR; Meta hasn't shipped Quest Link for Mac. Not near-term.

## Suggested first move once SSH'd in from Mac

1. **Read `Halloumi-Stream/README.md` and `Halloumi-Stream/docs/DESIGN.md`.** That's the streaming path we've already committed to. Anything you're about to do on the Mac should either advance it or explicitly not block it.
2. **Copy the brain bundle** (75 MB):
   ```
   scp nilkel@10.176.128.124:/home/nilkel/Projects/bitymi-demos/scenes/himalaya_brain/brain_2k.bitymi ~/Documents/Important/baked_atlas/
   ```
3. **Sanity-check WebGPU on your Mac** using the deployed viewer against the deployed bundle:
   `https://nilkel.github.io/bitymi-demos/viewer/index.html?bundle=https://huggingface.co/datasets/Nilkel/bitymi-demos/resolve/main/himalaya_brain/brain_2k.bitymi`
   If that renders cleanly, we know the Mac's Metal-via-WebGPU stack + our shaders + this specific bundle are all healthy. That's the render we're trying to match on the streaming path.
4. **Then decide** which Mac-side approach you want to iterate on:
   - Pure WebGPU (`Halloumi-WS` running locally via `npm run dev`) — fastest to see the brain rendering on the Mac. Zero new code.
   - Native Rust (`Halloumi-Stream/mac/`) — the actual streaming-ready path. Requires porting the compute pipeline from WGSL/TS-host to wgpu-rs.

Tell me which you want and I'll get more specific.
