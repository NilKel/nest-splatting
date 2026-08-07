# HIMALAYA / HiP-CT source data on KIT LSDF

Where the HIM-25 synchrotron CT volumes actually live, how to reach them, and how to
pull a usable downsampled volume without moving ~800 GB.

> This is the **primary** source. Lisa (`z0051bet`) also has copies of HIM-25-003/004/005
> on the sherlock cluster, but her home there is mode `700` with no ACL, so we cannot
> read them (see § 4). LSDF needs no favours from anyone.

---

## 1. Access

- **Host**: `os-login.lsdf.kit.edu` — a **3-node DNS round-robin** cluster
  (`lsdf-22-188-dis`, `-189-`, …). All nodes share one home, so a single
  `authorized_keys` covers all of them.
- **User**: `tum_txe9745` (this is the LDAP `localUid`; the TUM account federates in
  via bwIDM/Helmholtz AAI).
- **Home**: `/lsdf/tum/tum_tum/tum_txe9745` — i.e. `$LSDF` (`/lsdf`) + the LDAP
  `homeDir` field. The general layout is `/lsdf/<ORG>/<OU>/<user>`.
- **Auth**: SSH key. Password auth is also enabled, so plain `ssh-copy-id` works —
  no web-portal key upload needed despite what some SCC wiki pages imply.
- **Network**: reachable **directly from the workstation, no VPN** (141.52.212.83/84,
  port 22 open). Note the asymmetry: the `lsdf.kit.edu` **website** is Zscaler-blocked
  on the Siemens network ("Continuing Education/Colleges"), but SSH is not. Read the
  docs from a phone or off-network machine.

`~/.ssh/config` alias (installed):

```ssh
Host lsdf
    HostName os-login.lsdf.kit.edu
    User tum_txe9745
    IdentityFile ~/.ssh/id_ed25519
```

**Login-node etiquette**: LSDF's own docs say the login nodes are *not* for data
processing — only interactive login, file management, and data transfer. Do **not**
install a Python stack there and run conversions. The approach below keeps all
compute local (see § 3).

---

## 2. Data layout

Project group `scc-himalaya_p000357` → **`/lsdf/kit/scc/projects/himalaya`**.

```
/lsdf/kit/scc/projects/himalaya/
├── README.md                      # folder-structure + upload conventions
├── Poster/
└── Data/
    ├── data_avail.txt             # CSV: per-subject MB by modality — read this first
    ├── raw/<ID>/{ct,histo,mr3T,mr7T}/    # as-acquired, ct = JP2 slice stack
    ├── hdf5/CT_<ID>.h5                   # converted volumes + metadata_CT_<ID>.json
    └── Data/                             # "preliminary, unsorted, no longer maintained"
```

Subjects present in `raw/`: `FO335`, `FO336`, `HIM-25-001` … `HIM-25-015`.

### ⚠️ Two things that contradict the common assumption

1. **The HDF5 files contain NO downsampled pyramid.** `CT_HIM-25-011.h5` holds exactly
   one dataset, `HiP-CT/volume`, shape `(7964, 6732, 7061)` uint16 —
   **contiguous, unchunked, uncompressed** (757 GB, matching the file size exactly).
   The rest of the file is a `HiP-CT/metadata/**` tree of scalar groups. So the belief
   that "the h5 already has 16× downsampled versions as groups" does **not** hold for
   the files in `Data/hdf5/`. Any downsampled copies are someone's private derivatives.
2. **Only 4 subjects are converted to HDF5**: `011`, `013`, `014`, `015`. Everything
   else — including **HIM-25-005** — exists only as a raw JP2 slice stack.

### Sizes

| Subject | raw ct | raw histo | HDF5 |
|---|---|---|---|
| HIM-25-005 | 78 GB (8200 × JP2) | 82 GB | — (not converted) |
| HIM-25-011 | — | — | 757 GB |
| HIM-25-013 | — | — | 2105 GB |
| HIM-25-014 | — | — | 909 GB |
| HIM-25-015 | — | — | 947 GB |

HIM-25-005 CT: 8200 slices of **7604 × 6620 uint16** at **8.518 µm** isotropic
(`bm18` beamline, helical, 78 keV) ⇒ **826 GB uncompressed**, stored as ~10 MB JP2 per
slice. The prostate "overview" scan is the only CT series for this subject.

Workstation capacity for reference: `/` 467 GB (64 GB free), `/mnt/nilkel_hdd` 1.5 TB
(722 GB free). **A single full-resolution volume does not fit**, which is why § 3 exists.

---

## 3. Pulling a downsampled volume (the working recipe)

Mount read-only and do all decoding locally — LSDF only serves file reads:

```bash
mkdir -p ~/mnt/lsdf_himalaya
sshfs lsdf:/lsdf/kit/scc/projects/himalaya ~/mnt/lsdf_himalaya \
      -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,ro
```

Then `scripts/himalaya_downsample_ct.py` reads every Nth slice, block-mean reduces
in-plane by N, and writes an lzf-compressed `volume` dataset with `voxel_size_um`
attached:

```bash
conda run -n nest_splatting python scripts/himalaya_downsample_ct.py \
  --src ~/mnt/lsdf_himalaya/Data/raw/HIM-25-005/ct \
  --out data/himalaya/HIM-25-005/ct_ds16.h5 --factor 16
```

**Why every-Nth-slice matters**: measured sshfs throughput is only **~9 MB/s**, so the
full 78 GB stack would take ~2.5 h. At factor 16 we touch 513 of 8200 slices (~5 GB,
~10 min) and get a **513 × 413 × 475** uint16 volume at **136.3 µm** isotropic (~200 MB
uncompressed). In-plane is properly averaged; z is subsampled (averaging z would
require fetching every slice and defeat the point) — so expect some z aliasing.

Scaling: factor 8 → ~1.6 GB and ~20 min; factor 4 → ~13 GB and ~40 min. Both fit on
`/mnt/nilkel_hdd`. JP2 decode is the local bottleneck, not the network, past factor 8.

---

## 4. The sherlock alternative (blocked)

Lisa offered `/home/z0051bet/data` + `/home/z0051bet/projects/HIMALAYA` (which holds
the hdf5 conversion / downsampling / masking / cropping scripts, and Xingyu's
registration results for 004/005). `namei -l` shows the wall is her home directory
itself — `drwx------`, no ACL entries — so the grant never took effect. Both
`sherlock01h` (Princeton) and `sherlock_de_via_sshhop` (Erlangen login01) mount the
same quobyte storage, so neither route helps. To unblock, she needs the **traverse
bit**, which is the step usually missed:

```bash
setfacl -m u:z0051beu:--x /home/z0051bet          # execute-only: pass through, cannot list
setfacl -R -m u:z0051beu:rX /home/z0051bet/data /home/z0051bet/projects/HIMALAYA
setfacl -R -d -m u:z0051beu:rX /home/z0051bet/data
```

Still worth pursuing for the registration results and the processing scripts, but it
is no longer on the critical path for getting voxels.
