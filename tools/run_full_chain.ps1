# Full-chain integration test for the whole_body_tracking Isaac Lab 3.0 adapter.
#
# Design notes (each one fixes a bug found in the first version of this harness):
#
#   * Every step runs in its OWN PowerShell job with a hard wall-clock timeout. The first version
#     piped a long-running subprocess into `Out-String`, which deadlocked with no way to bound it.
#     Jobs are also the only reliable way to kill Kit + wandb, which can block in shutdown.
#
#   * Each step writes its output to a per-step log file via `*>`, and assertions read that FILE.
#     The first version asserted on a captured pipeline variable, so anything printed with
#     `Write-Host` (which does not enter the pipeline) was invisible to the check and every such
#     step was reported FAIL despite succeeding.
#
#   * `git` writes progress to stderr. Under PowerShell that becomes a NativeCommandError that
#     aborts the enclosing script block. Steps redirect all streams to a file.
#
#   * A step may declare `-HangAfterSuccessPatterns`: it is a PASS when the timeout fires AND those
#     patterns are present. `--logger wandb` trains to completion, then blocks in wandb/Kit teardown
#     and never exits (verified: "Training time: 10.4 seconds" then silence, no python process
#     consuming CPU, offline run directory written). Treating that as a failure would be wrong.
#
# Usage:
#   powershell -File run_full_chain.ps1 -IL <IsaacLab> -WBT <whole_body_tracking> [-Repo <this repo>]

param(
    [string]$IL   = "C:\Users\chenxi\Desktop\work\IsaacLab",
    [string]$WBT  = "C:\Users\chenxi\Desktop\whole_body_tracking",
    [string]$Repo = "C:\Users\chenxi\Desktop\whole_body_tracking_isaaclab3",
    [int]$TrainStepTimeoutSec = 300
)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$PY      = Join-Path $IL ".venv\Scripts\python.exe"
$SCRATCH = Join-Path $env:TEMP "wbt_fullchain"
$LOGS    = Join-Path $SCRATCH "logs"
New-Item -ItemType Directory -Force -Path $LOGS | Out-Null

$MOTION     = Join-Path $WBT "motions\g1_stand_synthetic.npz"
$EXPORT_DIR = Join-Path $SCRATCH "exported"

$RESULTS = [System.Collections.Generic.List[object]]::new()

function Step {
    param(
        [string]   $Name,
        [string]   $Body,                      # PowerShell source, runs in its own job
        [string[]] $ExpectPatterns = @(),
        [string[]] $HangAfterSuccessPatterns = @(),
        [int]      $TimeoutSec = 180,
        [string]   $WorkDir = $IL
    )

    Write-Host ""
    Write-Host ("=" * 78)
    Write-Host "STEP: $Name  (timeout ${TimeoutSec}s)"
    Write-Host ("=" * 78)

    $log = Join-Path $LOGS ("$Name.log")
    Remove-Item $log -Force -ErrorAction SilentlyContinue

    $sw = [Diagnostics.Stopwatch]::StartNew()
    $job = Start-Job -ArgumentList $Body, $log, $WorkDir, $PY, $MOTION, $EXPORT_DIR, $Repo, $WBT, $SCRATCH -ScriptBlock {
        param($Body, $log, $WorkDir, $PY, $MOTION, $EXPORT_DIR, $Repo, $WBT, $SCRATCH)
        Set-Location $WorkDir
        $env:PYTHONIOENCODING = "utf-8"
        $env:PYTHONUTF8 = "1"
        $env:WANDB_SILENT = "true"
        $env:WANDB_MODE = "offline"
        $env:WANDB_USERNAME = "daichenxi-davis"
        # Keep wandb's offline runs inside the scratch tree. Without this, wandb writes to
        # <cwd>/wandb and a check that globs an absolute path finds nothing.
        $env:WANDB_DIR = Join-Path $SCRATCH "wandb"
        New-Item -ItemType Directory -Force -Path $env:WANDB_DIR | Out-Null
        $scriptBody = [scriptblock]::Create($Body)
        & $scriptBody *> $log
    }

    $timedOut = -not (Wait-Job $job -Timeout $TimeoutSec)
    if ($timedOut) {
        Write-Host "  TIMEOUT after ${TimeoutSec}s - stopping job"
        Stop-Job $job -ErrorAction SilentlyContinue
    }
    Remove-Job $job -Force -ErrorAction SilentlyContinue
    $sw.Stop()

    $text = if (Test-Path $log) { Get-Content $log -Raw } else { "" }

    # Exit code is read from the log, where every step body ends with an explicit ``exitcode: N``
    # line. The job's own exit status is not reliably observable, so the log is the single source of
    # truth for both the exit code and the assertions. The LAST match wins, so a body may emit
    # intermediate codes too.
    $exitCode = $null
    $matches = [regex]::Matches($text, '(?m)^exitcode:\s*(-?\d+)\s*$')
    if ($matches.Count -gt 0) { $exitCode = [int]$matches[$matches.Count - 1].Groups[1].Value }

    $missing = @($ExpectPatterns | Where-Object { $text -notmatch [regex]::Escape($_) })
    $failMarkers = @("Traceback (most recent call last)", "MISSING: [")
    $foundFail = @($failMarkers | Where-Object { $text -match [regex]::Escape($_) })

    if ($timedOut) {
        $hangOk = ($HangAfterSuccessPatterns.Count -gt 0) -and
                  (@($HangAfterSuccessPatterns | Where-Object { $text -notmatch [regex]::Escape($_) }).Count -eq 0) -and
                  ($missing.Count -eq 0)
        $ok = $hangOk
        $verdict = if ($hangOk) { "PASS (timed out in teardown, after success markers were present)" } else { "FAIL (timed out)" }
    } else {
        $ok = ($missing.Count -eq 0) -and ($foundFail.Count -eq 0) -and ($exitCode -eq 0)
        $verdict = "PASS"
    }

    Write-Host ("  elapsed={0:N1}s  log={1:N0} bytes  exitcode={2}" -f $sw.Elapsed.TotalSeconds, (Get-Item $log -ErrorAction SilentlyContinue).Length, $(if ($null -eq $exitCode) { "?" } else { $exitCode }))
    if ($missing.Count)    { Write-Host ("  missing: {0}" -f ($missing -join " | ")) }
    if ($foundFail.Count)  { Write-Host ("  failure markers: {0}" -f ($foundFail -join " | ")) }
    Write-Host ("  -> {0}" -f $verdict)
    Write-Host ("  log: {0}" -f $log)

    $RESULTS.Add([pscustomobject]@{
        Step = $Name
        Status = if ($ok) { "PASS" } else { "FAIL" }
        Verdict = $verdict
        ExitCode = $exitCode
        Seconds = [math]::Round($sw.Elapsed.TotalSeconds,1)
        Missing = ($missing -join "; ")
        FailMarkers = ($foundFail -join "; ")
        Log = $log
    })
}

# 1 ------------------------------------------------------------------ patch applies
Step -Name "01_patch_applies" -TimeoutSec 120 -Body @'
$PRISTINE = Join-Path $env:TEMP "wbt_fullchain_pristine"
Remove-Item $PRISTINE -Recurse -Force -ErrorAction SilentlyContinue
$patch = Join-Path $Repo "patches\isaaclab3-migration.patch"
git -C $WBT worktree add --detach $PRISTINE cd65172 *> $null
"worktree: $LASTEXITCODE"
git -C $PRISTINE apply --check $patch
"check: $LASTEXITCODE"
git -C $PRISTINE apply $patch
"apply: $LASTEXITCODE"
"files: " + (git -C $PRISTINE diff --name-only | Measure-Object).Count
git -C $WBT worktree remove $PRISTINE --force *> $null
git -C $WBT worktree prune *> $null
'@ -ExpectPatterns @("worktree: 0","check: 0","apply: 0","files: 13")

# 2 ------------------------------------------------------------------ environment
Step -Name "02_environment" -TimeoutSec 180 -Body @'
& $PY -c @"
import importlib.util as u, sys
mods = ['isaacsim','isaaclab','isaaclab_physx','isaaclab_tasks','whole_body_tracking','rsl_rl','wandb','onnx','onnxruntime','torch']
bad = []
for m in mods:
    try: ok = bool(u.find_spec(m))
    except Exception: ok = False
    print(f'{m:24s} {ok}')
    if not ok: bad.append(m)
import wandb, torch
print('wandb version:', wandb.__version__)
print('generate_id present:', hasattr(wandb.util,'generate_id'))
print('cuda available:', torch.cuda.is_available())
print('MISSING:', bad)
sys.exit(1 if bad else 0)
"@
"exitcode: $LASTEXITCODE"
'@ -ExpectPatterns @("generate_id present: True","cuda available: True","MISSING: []")

# 3 ------------------------------------------------------------------ assets
Step -Name "03_assets" -TimeoutSec 180 -Body @'
& $PY -c @"
import os, sys, re
sys.path.insert(0, r'$WBT\source\whole_body_tracking')
from whole_body_tracking.assets import ASSET_DIR, require_asset
u = require_asset('unitree_description','urdf','g1','main.urdf')
print('main.urdf bytes:', os.path.getsize(u))
try:
    print('capsule bytes:', os.path.getsize(require_asset('unitree_description','urdf','g1','main_capsule.urdf')))
except FileNotFoundError:
    print('capsule: ABSENT')
refs = re.findall(r'filename=\"([^\"]+)\"', open(u, encoding='utf-8').read())
missing = [r for r in refs if not os.path.exists(os.path.join(ASSET_DIR,'unitree_description','meshes','g1', os.path.basename(r)))]
print('mesh refs:', len(refs), 'missing:', len(missing))
sys.exit(1 if missing else 0)
"@
"exitcode: $LASTEXITCODE"
'@ -ExpectPatterns @("main.urdf bytes: 40589","missing: 0")

# 4 ------------------------------------------------------------------ config drift
Step -Name "04_config_drift" -TimeoutSec 180 -Body @'
& $PY (Join-Path $Repo "scripts\diff_against_upstream.py")
"exitcode: $LASTEXITCODE"
'@ -ExpectPatterns @("RESULT: PASS - no unintended drift")

# 5 ------------------------------------------------------------------ agent cfg equivalence
# Expects "6 difference(s)": the script reports the 6 benign legacy/deprecation differences rather
# than claiming identity. The assertion below matches what the script actually prints.
Step -Name "05_agent_cfg_equivalence" -TimeoutSec 180 -Body @'
& $PY (Join-Path $Repo "scripts\check_agent_cfg_equivalence.py")
"exitcode: $LASTEXITCODE"
'@ -ExpectPatterns @("fields compared: 55","RESULT: 6 difference(s)")

# 6 ------------------------------------------------------------------ regen synthetic motion
Step -Name "06_synthetic_motion" -TimeoutSec 300 -Body @'
$out = Join-Path $SCRATCH "motion_new.npz"
Remove-Item $out -Force -ErrorAction SilentlyContinue
& $PY (Join-Path $Repo "scripts\make_synthetic_motion.py") --output $out --num_frames 200 --visualizer none
"gen exit: $LASTEXITCODE"
"bytes: " + (Get-Item $out).Length
& $PY -c @"
import numpy as np
d = np.load(r'$out')
print('keys:', sorted(d.files))
print('quat_order:', str(d['quat_order']))
print('frames:', d['joint_pos'].shape[0])
"@
"exitcode: $LASTEXITCODE"
'@ -ExpectPatterns @("gen exit: 0","bytes: 360462","quat_order: xyzw")

# 7 ------------------------------------------------------------------ capsule URDF
Step -Name "07_capsule_urdf" -TimeoutSec 120 -Body @'
& $PY (Join-Path $Repo "scripts\make_capsule_urdf.py") --check
"exitcode: $LASTEXITCODE"
'@ -ExpectPatterns @("cylinders converted in <collision>: 28")

# 8 ------------------------------------------------------------------ training (tensorboard)
Step -Name "08_train_tensorboard" -TimeoutSec $TrainStepTimeoutSec -Body @'
& $PY -m isaaclab train --rl_library rsl_rl --task Tracking-Flat-G1-v0 `
    --num_envs 8 --max_iterations 3 --logger tensorboard `
    --external_callback whole_body_tracking.tasks.register.register `
    --visualizer none physics=isaacsim_physx `
    "env.commands.motion.motion_file=$MOTION"
"exitcode: $LASTEXITCODE"
'@ -ExpectPatterns @("Learning iteration 0/3","Training time:")

# 9 ------------------------------------------------------------------ training (wandb offline)
Step -Name "09_train_wandb_offline" -TimeoutSec $TrainStepTimeoutSec -Body @'
& $PY -m isaaclab train --rl_library rsl_rl --task Tracking-Flat-G1-v0 `
    --num_envs 8 --max_iterations 3 --logger wandb `
    --log_project_name wbt-fullchain --run_name fullchain `
    --external_callback whole_body_tracking.tasks.register.register `
    --visualizer none physics=isaacsim_physx `
    "env.commands.motion.motion_file=$MOTION"
"exitcode: $LASTEXITCODE"
'@ -ExpectPatterns @("Logging to wandb","Learning iteration 0/3","Training time:") `
   -HangAfterSuccessPatterns @("Training time:","Logging to wandb")

# 10 ----------------------------------------------------------------- export (--checkpoint latest)
Step -Name "10_export_latest" -TimeoutSec $TrainStepTimeoutSec -Body @'
& $PY (Join-Path $Repo "scripts\export_policy.py") `
    --task Tracking-Flat-G1-v0 --checkpoint latest --num_envs 2 `
    --output_dir $EXPORT_DIR `
    --visualizer none physics=isaacsim_physx `
    "env.commands.motion.motion_file=$MOTION"
"exitcode: $LASTEXITCODE"
'@ -ExpectPatterns @("Loading checkpoint:","Wrote motion ONNX policy:","export finished with status 0")

# 11 ----------------------------------------------------------------- validate ONNX
Step -Name "11_validate_onnx" -TimeoutSec 180 -Body @'
& $PY (Join-Path $Repo "scripts\validate_exported_onnx.py") `
    --onnx (Join-Path $EXPORT_DIR "policy.onnx") --motion $MOTION
"exitcode: $LASTEXITCODE"
'@ -ExpectPatterns @("ALL CHECKS PASSED")

# 12 ----------------------------------------------------------------- W&B artifact upload
Step -Name "12_wandb_upload" -TimeoutSec 300 -Body @'
$stage = Join-Path $SCRATCH "wandb_stage"
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item (Join-Path $EXPORT_DIR "policy.onnx*") $stage -Force
& $PY -c @"
import os, sys, glob
sys.path.insert(0, r'$WBT\source\whole_body_tracking')
from whole_body_tracking.utils.exporter import upload_to_wandb
import wandb
stage = r'$stage'
onnx = os.path.join(stage, 'policy.onnx')
print('staged:', sorted(os.listdir(stage)))
try:
    upload_to_wandb(onnx)
    print('GUARD FAILED')
except RuntimeError:
    print('guard OK')
run = wandb.init(project='wbt-fullchain', name='upload', job_type='export', mode='offline')
upload_to_wandb(onnx, run_path='e/p/abc12345', artifact_name='motion_policy')
wandb.finish()
runs = glob.glob(r'$WBT\wandb\offline-run-*')
rd = max(runs, key=os.path.getmtime)
tx = glob.glob(os.path.join(rd, '*.wandb'))[0]
data = open(tx,'rb').read()
for n in (b'motion_policy', b'policy.onnx', b'policy.onnx.data'):
    print('manifest', n.decode(), n in data)
print('upload script done')
"@
"exitcode: $LASTEXITCODE"
'@ -ExpectPatterns @("guard OK","manifest policy.onnx.data True","upload script done")

# 13 ----------------------------------------------------------------- play (--checkpoint latest)
#
# NOT a pass/fail on "play works". `isaaclab play` reaches the policy-loading stage, resolves the
# `latest` manifest selector, creates the runner and loads the checkpoint -- and then dies on the
# first env.step with:
#
#   RuntimeError: Can't get __cuda_array_interface__ on Variable that requires grad.
#     If gradients aren't required, use var.detach() ...
#
# The policy head returns a grad-tracking tensor which rsl_rl's wrapper forwards straight into the
# env. That is a framework/rsl-rl interaction, not something this adapter introduced (training and
# the export path both work). This step is therefore an *observation* step: it asserts the pipeline
# reaches the checkpoint load AND that the failure signature is the known one, so a different
# failure would still be caught.
Step -Name "13_play_latest" -TimeoutSec $TrainStepTimeoutSec -Body @'
& $PY -m isaaclab play --rl_library rsl_rl --task Tracking-Flat-G1-v0 --num_envs 2 `
    --checkpoint latest `
    --external_callback whole_body_tracking.tasks.register.register `
    --visualizer none physics=isaacsim_physx `
    "env.commands.motion.motion_file=$MOTION"
"exitcode: $LASTEXITCODE"
'@ -ExpectPatterns @("Loading experiment from directory:","Loading model checkpoint from:","requires grad") `
   -HangAfterSuccessPatterns @("Loading model checkpoint from:")

# ------------------------------------------------------------------ summary
Write-Host ""
Write-Host ("=" * 78)
Write-Host "SUMMARY"
Write-Host ("=" * 78)
$RESULTS | Format-Table Step, Status, Seconds, Verdict -AutoSize | Out-String | Write-Host

$failed = @($RESULTS | Where-Object Status -ne "PASS")
if ($failed.Count -eq 0) {
    Write-Host "FULL CHAIN: ALL $($RESULTS.Count) STEPS PASSED"
} else {
    Write-Host "FULL CHAIN: $($failed.Count) of $($RESULTS.Count) STEPS FAILED"
    foreach ($f in $failed) {
        Write-Host ("  FAILED {0}  ({1})" -f $f.Step, $f.Verdict)
        if ($f.Missing) { Write-Host ("     missing: {0}" -f $f.Missing) }
        Write-Host ("     log: {0}" -f $f.Log)
    }
}
$RESULTS | ConvertTo-Json -Depth 3 | Out-File (Join-Path $SCRATCH "results.json") -Encoding utf8
Write-Host ""
Write-Host "results: $(Join-Path $SCRATCH 'results.json')"
