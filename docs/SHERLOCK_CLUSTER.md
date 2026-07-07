# Sherlock Cluster (`sherlock01.ainet.local`)

Compute cluster used for paper-scale runs (A100 / job-array sweeps). Distinct
from the 4090 dev box documented in `BENCH_4090.md`.

> Connection from `nilkel-Workstation` (Linux, `10.176.128.124`) has been
> verified end-to-end via the office-direct IP `10.80.42.75`. The MacBook
> side uses the FQDN `sherlock01.ainet.local`; that DNS suffix does **not**
> resolve from this workstation's network, so the alias here maps to the IP
> instead. Items that come from indirect evidence (slurm scripts in
> `../beta-splatting/`, VSCode logs) and have not been re-verified on a live
> shell are still flagged **[ASSUMED]**.

---

## 0. Login node etiquette — NEVER run work on the login node

**Rule:** The login node is shared infrastructure for hundreds of users. We
do **everything** on a compute node via `srun` (or `salloc + srun --overlap`)
— including bulk file ops, packing/unpacking archives, rsync between project
dirs, conda installs, builds, `du`/`find` over large trees, and any one-off
Python invocation. The login node is **only** for:

- Submitting jobs (`sbatch`, `squeue`, `scancel`, `scontrol`)
- Editing small files and inspecting log tails
- Lightweight `ls` / `cat` of small files

Everything else gets a quick allocation first. Typical pattern for ad-hoc
work (e.g. zipping outputs, moving data, running a Python aggregation
script):

```bash
# One-shot interactive shell on a compute node:
srun --partition=a100-4gpu-40gb --account=rctcd82061 \
     --gres=gpu:1 --time=01:00:00 --pty bash
# … then do everything inside this shell.
```

For longer or scripted ad-hoc work, allocate once and reuse via `--overlap`:

```bash
salloc --no-shell --partition=a100-4gpu-40gb --account=rctcd82061 \
       --gres=gpu:1 --time=4:00:00 --job-name=adhoc
# pick up the JOBID from sacct/squeue, then:
srun --jobid=$JOBID --overlap bash -lc "<command>"
```

If a task doesn't need a GPU, drop `--gres=gpu:1` to share node resources
more cheaply. The principle is the same: **the login node never runs the
real work.**

### 0a. Automation / agents: do NOT spam the login node with `ssh`+`srun`

This bit us hard (2026-06-22). Driving the cluster with **dozens of
`ssh sherlock01h '... srun ... bash ...'` calls** — many wrapped in `timeout`
or later `scancel`led — plus a monitor that `ssh`-polled every 5 min, **orphans
the remote `srun` + `bash`** on the login node each time a call is killed or
times out. They pile up under your user, hit the **per-user process cap**, and
the login node starts returning:

```
/usr/libexec/grepconf.sh: fork: retry: Resource temporarily unavailable   # shell rc can't fork
Connection to <ip> closed by remote host                                  # sshd can't spawn your session
```

Once wedged you **can't even get a shell to clean up** (kill/ps also need to
fork). It's a *per-user* cap, so other users are fine — but it locks *you* out
of the login node. (`ping` may still answer while TCP/22 sessions die — that
combo = login node out of process slots, NOT a network or disk problem.)

**Rules to avoid it:**
- **One long-lived session, not many short ones.** `salloc --no-shell …` once,
  then `srun --jobid=$JOBID --overlap bash -lc "<all the work>"`; or a single
  `ssh` running a script that does everything. Never loop `ssh host 'srun …'`.
- **Don't wrap `ssh … srun …` in tight `timeout`s** or `scancel` them abruptly —
  that's what orphans the remote side. `scancel` the *job id*; don't kill the
  ssh client out from under a running `srun`.
- **No polling monitors over `ssh`.** Batch status into one infrequent call
  (`ssh host 'squeue; sacct; tail …'`), not a per-minute loop.
- `sbatch`/`squeue`/`sacct` in moderation are cheap; repeated session/`srun`
  **spawns** are the hazard.

**Recovery when already wedged:**
- Single-fork kill (retry a few times — needs only one fork to land):
  `ssh sherlock01h "pkill -9 -u z0051beu -f srun"`
- If SSH returns `Connection closed by remote host` (can't spawn a session):
  **wait 15–30 min** — orphaned timed-out `ssh`/`srun`/`sleep` clients die on
  their own and free fork slots — then retry the `pkill` **once**. Don't keep
  hammering it; each attempt adds churn.
- Else **HPC support**: "login node sherlock01, user z0051beu hitting its
  process limit (`fork: Resource temporarily unavailable`); please clear my
  orphaned `srun`/`bash` or bounce the login node." Fixed in seconds.
- **SBATCH jobs are unaffected** — they run on compute nodes; login-node fork
  exhaustion never touches them.

---

## 1. Connection

- **Hostname (MacBook side)**: `sherlock01.ainet.local` (AInet internal DNS)
- **IP (Princeton Office direct)**: `10.80.42.75` (MacBook config calls this `sherlock01o`)
- **User**: `z0051beu`
- **Account (SLURM `--account`)**: `rctcd82061`
- **Network**:
  - From this workstation (`10.176.128.x`): **`.ainet.local` does NOT resolve.**
    Use the `10.80.42.75` IP directly (TCP confirmed open on port 22).
  - From a MacBook on the org network / VPN: the FQDN resolves and `sherlock01h`
    works as configured in your `~/.ssh/config`.
- **Auth**: SSH key only. This workstation's `~/.ssh/id_ed25519.pub` is in
  `z0051beu@sherlock01:~/.ssh/authorized_keys`; no password fallback needed
  for routine commands. (Fallback: SSO/LDAP password if you ever need a fresh
  enrollment.)

### `~/.ssh/config` blocks

**On this workstation** (`/home/nilkel/.ssh/config` — already installed):

```ssh
# Princeton Office cluster sherlock01.
# DNS suffix .ainet.local doesn't resolve from this workstation's network;
# point the alias straight at the office-direct IP (= MacBook's `sherlock01o`).
Host sherlock01h
    HostName 10.80.42.75
    User z0051beu
    IdentityFile ~/.ssh/id_ed25519

Host *
    ServerAliveInterval 60
    ServerAliveCountMax 3
    ConnectTimeout 0
    ConnectionAttempts 10
```

**On a MacBook** (reference — uses the FQDN, plus all the other AInet routes):

```ssh
Host sherlock01h
    HostName sherlock01.ainet.local
    User z0051beu
    IdentityFile ~/.ssh/id_ed25519
```

### Verification

```bash
ssh -o BatchMode=yes sherlock01h hostname
# → sherlock01
```

### Initial key installation (one-shot, already done for this workstation)

From the source host (the box whose key you're installing):

```bash
# From a machine that already has password / key access (e.g. MacBook):
ssh sherlock01h 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
                 echo "<source-machine-pubkey>" >> ~/.ssh/authorized_keys && \
                 chmod 600 ~/.ssh/authorized_keys && echo OK'
```

For this workstation that was:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEnn7CM5opQELNisOjUxOdHlEV/NDMDTTHkx4IF60asS nilkel@nilkel-Workstation
```

### When auth breaks ("VSCode keeps asking for password")

Symptom from a VSCode Remote-SSH log:
```
> z0051beu@sherlock01.ainet.local's password:
> Permission denied, please try again.
```
Diagnostic order:

1. `ssh -v sherlock01h` from a plain terminal. Look for `Offering public key:` —
   if absent, your **local** side never sent a key.
2. `ssh-add -l` — agent has the key? If empty: `ssh-add ~/.ssh/id_ed25519`.
3. If key is offered but rejected, the problem is on the cluster:
   `~/` must be mode `750` or stricter; `~/.ssh` `700`; `~/.ssh/authorized_keys` `600`.
4. Stale VSCode server on the host? Command palette →
   *Remote-SSH: Kill VS Code Server on Host…* then reconnect.
5. Lost SSO password → org self-service portal or HPC support.

---

## 2. Filesystem layout

Two relevant trees, from `slurm_benchmark_mip360.sh`:

```
~/userdir/                                  # user-managed area (NOT $HOME directly)
├── miniconda3/                             # conda install (sourced from here)
│   └── etc/profile.d/conda.sh
└── Projects/
    ├── beta-splatting/                     # repo
    │   ├── slurm_benchmark_mip360.sh
    │   ├── slurm_logs/                     # SLURM stdout/stderr land here
    │   └── eval_mip360/<scene>/            # outputs
    └── data/
        └── mip_360/<scene>/                # COLMAP scene roots
```

- **`~/userdir/`** is the convention on this cluster — `$HOME` itself appears
  to be small/quota-limited; bulk projects + conda envs live under `userdir/`. **[ASSUMED]**
- **Scratch / fast scratch**: not observed in any script. Check cluster docs;
  typically `/scratch/<user>/` or similar on HPC sites.
- **Output discipline**: scripts write to `./eval_*/` relative to the project
  directory rather than to absolute paths, so the working dir matters.

---

## 3. SLURM job pattern

Template from `../beta-splatting/slurm_benchmark_mip360.sh` (verified working):

```bash
#!/bin/bash
#SBATCH --partition=a100-4gpu-40gb         # see § Partitions
#SBATCH --account=rctcd82061
#SBATCH --gres=gpu:1                       # one GPU per task
#SBATCH --job-name=<job>
#SBATCH --output=slurm_logs/<job>_%a_%j.out
#SBATCH --error=slurm_logs/<job>_%a_%j.err
#SBATCH --time=24:00:00
#SBATCH --array=1-N                        # job array for sweeps

set -e

module load gcc/13.1.0          # gcc/13.2.0 no longer present
# NOTE: do NOT `module load cuda12.1/...` — that module was removed (only 13.0
# exists). For nest_splatting (torch cu121) the CUDA 12.1 toolkit lives in the
# conda env (installed from conda-forge, see § Modules); point at it instead:
source ~/userdir/miniconda3/etc/profile.d/conda.sh
conda activate <env-name>
export CUDA_HOME=$CONDA_PREFIX; export PATH=$CONDA_PREFIX/bin:$PATH  # nvcc 12.1 (for builds)

cd ~/userdir/Projects/<repo>
mkdir -p slurm_logs

# ... job body ...
```

### Submission

```bash
ssh sherlock
cd ~/userdir/Projects/<repo>
sbatch slurm_benchmark_mip360.sh                 # all array tasks
sbatch --array=1-4  slurm_benchmark_mip360.sh    # subset
squeue -u $USER                                   # check queue
scancel <job_id>                                  # cancel
```

### Partitions observed

| Partition | Verified | Notes |
|---|---|---|
| `a100-4gpu-40gb` | ✓ | A100, 40 GB. 4 GPUs/node; request 1 with `--gres=gpu:1`. |
| _others_ | — | Not seen in this repo. `sinfo` on the cluster to enumerate. |

### Modules

- `gcc/13.1.0` — compiler for CUDA extensions (loaded by default; `gcc/13.2.0`
  is no longer present, `gcc/15.1.0`/`gcc11/11.5.0` also available).
- **⚠️ `cuda12.1/toolkit/12.1.0` HAS BEEN REMOVED** (verified 2026-06-22).
  `module avail` now shows only **`cuda13.0/toolkit/13.0`** (shared apps:
  `cuda11.7`, `cuda13.0`, `cuda13.2U1`; system `/usr/local/cuda` = 13.x). The
  `nest_splatting` env's torch is **cu121**, so building its CUDA extensions
  needs **nvcc 12.1** — a 13.x toolkit fails with
  `_check_cuda_version` "detected 13.x vs PyTorch 12.1".

> **Get CUDA 12.1 on the cluster (modules can't anymore): install it into the
> conda env from conda-forge.** The `nvidia` channel's pins do NOT propagate to
> the compiler (`cuda-nvcc` stays 13.x); conda-forge's `cuda-version` metapackage
> is the canonical global pin:
> ```bash
> conda install -y --solver=libmamba --override-channels \
>   -c https://prefix.dev/conda-forge \
>   cuda-version=12.1 cuda-toolkit=12.1.1 cuda-nvcc=12.1.105
> # then build (on a COMPUTE node):
> export CUDA_HOME=$CONDA_PREFIX; export PATH=$CONDA_PREFIX/bin:$PATH
> module load gcc/13.1.0; export TORCH_CUDA_ARCH_LIST=8.0   # A100 sm_80
> rm -rf <submod>/build <submod>/*/_C.cpython-310*.so       # force clean recompile
> cd <submod> && python -m pip install -e . --no-build-isolation
> ```
> Verify `nvcc --version | grep release` == 12.1 BEFORE compiling. This toolkit
> install persists in the env, so it's a one-time fix.

> 5090 / Blackwell note: the 5090 forces CUDA 12.8 + PyTorch nightly cu128, so
> its prebuilt `.so`s won't load on the A100s (sm_80, cu121). Always rebuild the
> CUDA submodules on sherlock against the env's CUDA 12.1 (installed as above).
> See § Common pitfalls.

---

## 4. Conda environments

```bash
source ~/userdir/miniconda3/etc/profile.d/conda.sh
conda activate <env>
```

Envs observed:
- `beta_splatting` (from `slurm_benchmark_mip360.sh`).
- Others: unknown — `conda env list` on the cluster will enumerate.

For nest-splatting, mirror the env-bootstrap from `bench_4090.md` /
`.claude/rules/nest-splatting.md`. **[ASSUMED]** no pre-existing
`nest_splatting` env on sherlock; create with the same recipe but against
`cuda12.1/toolkit` (the module-loaded version) instead of system CUDA.

---

## 5. Typical experiment lifecycle

1. **Sync code** from local box:
   ```bash
   rsync -av --exclude '.git' --exclude 'eval*/' \
     ~/Projects/<repo>/ sherlock:~/userdir/Projects/<repo>/
   ```
2. **Sync data** (one-shot, large):
   ```bash
   rsync -av ~/Projects/<repo>/data/<dataset>/ \
     sherlock:~/userdir/Projects/data/<dataset>/
   ```
3. **Submit**: `ssh sherlock 'cd ~/userdir/Projects/<repo> && sbatch slurm_*.sh'`.
4. **Monitor**: `ssh sherlock 'squeue -u z0051beu; tail -n50 ~/userdir/Projects/<repo>/slurm_logs/<job>_*_<id>.out'`.
5. **Pull results**:
   ```bash
   rsync -av sherlock:~/userdir/Projects/<repo>/eval_*/ \
     ~/Projects/<repo>/eval_remote/
   ```

---

## 6. Common pitfalls

- **VPN / network**: `*.ainet.local` is internal — connection failures usually
  mean VPN is down, not a credential problem.
- **`~` perms**: SSH silently rejects pubkeys if `~` is group/world-writable.
  `chmod go-w ~` after any home-dir-touching script.
- **CUDA arch / version mismatch**: A100 = sm_80; the local 5090 (sm_120) needs
  CUDA 12.8 + cu128 PyTorch, so its prebuilt `.so`s won't load on sherlock.
  Rebuild CUDA extensions on the cluster — but note the **`cuda12.1` module is
  gone** (only `cuda13.0` exists), and the env's torch is cu121, so a 13.x nvcc
  fails the torch version check. Install CUDA 12.1 into the env from conda-forge
  (`cuda-version=12.1`, see § Modules) and build with `CUDA_HOME=$CONDA_PREFIX`,
  `TORCH_CUDA_ARCH_LIST=8.0`. Which submodule to rebuild matters: `cat`/`baseline`
  use `diff-surfel-rasterization` (not `diff_surfel_3D`); `3D_SH_res`/`mixed`/etc.
  use their own. A stale `.so` (older than its `.cu`) silently runs old code.
- **Relative paths in slurm scripts**: SLURM doesn't change cwd; the body
  must `cd ~/userdir/Projects/<repo>` before invoking anything.
- **Resuming from `metrics.json`**: `slurm_benchmark_mip360.sh` skips scenes
  whose `iteration_best/metrics.json` already exists. Delete that file (or
  the whole `iteration_best/`) to force a re-run.
- **`--array` task ID quirk**: array task IDs are 1-indexed in this script,
  so `--array=1-9` runs all nine mip-360 scenes; `--array=4` runs just bonsai.

---

## 7. References

- `../beta-splatting/slurm_benchmark_mip360.sh` — the canonical working slurm template.
- `BENCH_4090.md` — companion doc for the 4090 dev box (`neel@10.176.128.69`).
- Org HPC support — for password resets, partition / quota docs that aren't
  in any committed file.
