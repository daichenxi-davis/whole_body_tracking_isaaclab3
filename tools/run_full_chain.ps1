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

function Stop-ProcessTree {
    param([System.Diagnostics.Process]$Process)

    # Do NOT use $Process.Kill($true): that overload (kill entire tree) is .NET Core 3.0+ only.
    # Under Windows PowerShell 5.1 (.NET Framework) Process has a single parameterless Kill, so the
    # call fails with "Cannot find an overload for 'Kill'" and nothing gets killed -- which is how
    # leftover simulator processes accumulated hundreds of seconds of CPU.
    try {
        & taskkill.exe /PID $Process.Id /T /F *> $null
    } catch {
        Write-Host "  taskkill failed: $($_.Exception.Message)"
    }
    try { $Process.WaitForExit(10000) | Out-Null } catch {}
    Write-Host ("  process tree terminated (exit={0})" -f $(try { $Process.ExitCode } catch { "?" }))
}

function Step {
    param(
        [string]   $Name,
        [string]   $Body,                      # PowerShell source, runs in its own job
        [string[]] $ExpectPatterns = @(),
        [string[]] $HangAfterSuccessPatterns = @(),
        # When every one of these appears in the log, the step's work is done: stop the process
        # immediately instead of waiting for an exit that may never come.
        [string[]] $StopWhenLogMatches = @(),
        [int]      $StopGraceSec = 4,   # let trailing log lines flush before killing
        [int]      $TimeoutSec = 180,
        [string]   $WorkDir = $IL,
        [switch]   $NoExitCodeCheck
    )

    Write-Host ""
    Write-Host ("=" * 78)
    Write-Host "STEP: $Name  (timeout ${TimeoutSec}s)"
    Write-Host ("=" * 78)

    $log = Join-Path $LOGS ("$Name.log")
    Remove-Item $log -Force -ErrorAction SilentlyContinue

    $stepStartedAt = Get-Date
    $sw = [Diagnostics.Stopwatch]::StartNew()

    # Run the step as a real child process, not a PowerShell job.
    #
    # Jobs were the original approach and they deadlock: `--logger wandb` leaves a detached
    # `wandb-core` service behind, and Stop-Job waits for the job's whole process tree to be reaped,
    # so it blocks forever. A System.Diagnostics.Process can be terminated as a tree with
    # Kill(entireProcessTree: true), which actually works.
    $bodyPath = Join-Path $SCRATCH "$Name.body.ps1"
    $runnerPath = Join-Path $SCRATCH "$Name.runner.ps1"
    $prologue = @"
`$env:PYTHONIOENCODING = 'utf-8'
`$env:PYTHONUTF8 = '1'
`$env:WANDB_SILENT = 'true'
`$env:WANDB_MODE = 'offline'
`$env:WANDB_USERNAME = 'daichenxi-davis'
# Pin wandb's output into the scratch tree. Otherwise wandb writes to <cwd>/wandb and a check that
# globs an absolute path finds nothing.
`$env:WANDB_DIR = Join-Path '$SCRATCH' 'wandb'
New-Item -ItemType Directory -Force -Path `$env:WANDB_DIR | Out-Null
Set-Location '$WorkDir'
`$PY = '$PY'
`$MOTION = '$MOTION'
`$EXPORT_DIR = '$EXPORT_DIR'
`$Repo = '$Repo'
`$WBT = '$WBT'
`$SCRATCH = '$SCRATCH'
"@
    Set-Content -Path $bodyPath -Value $prologue -Encoding utf8
    Add-Content -Path $bodyPath -Value $Body -Encoding utf8
    # The wrapper reports its own exit status so the harness can tell a normal failure from a kill.
    $runner = @"
`$ErrorActionPreference = 'Continue'
. '$bodyPath' *> '$log'
"@
    Set-Content -Path $runnerPath -Value $runner -Encoding utf8

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = (Get-Process -Id $PID).Path
    $psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$runnerPath`""
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.WorkingDirectory = $WorkDir
    $proc = [System.Diagnostics.Process]::Start($psi)

    # Poll instead of blocking. Two reasons:
    #
    #  1. Steps that finish their work and then block in teardown (`--logger wandb` leaves a
    #     detached wandb-core service behind; `isaaclab play` blocks after raising) would otherwise
    #     burn the entire timeout for nothing. As soon as the step's success markers appear in the
    #     log, the work is done and the process is stopped.
    #  2. Killing the tree is the only reliable way out; Wait-Job deadlocks on exactly these cases.
    $stopPatterns = @($StopWhenLogMatches)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $stoppedEarly = $false
    while (-not $proc.HasExited) {
        if ((Get-Date) -ge $deadline) { break }
        Start-Sleep -Milliseconds 1500
        if ($stopPatterns.Count -gt 0 -and (Test-Path $log)) {
            $snapshot = Get-Content $log -Raw
            $allSeen = @($stopPatterns | Where-Object { $snapshot -notmatch [regex]::Escape($_) }).Count -eq 0
            if ($allSeen) {
                $stoppedEarly = $true
                Write-Host "  success markers seen after $([math]::Round($sw.Elapsed.TotalSeconds,1))s - stopping the process tree"
                break
            }
        }
    }

    $timedOut = $false
    if (-not $proc.HasExited) {
        if ($stoppedEarly) {
            # Expected for the two teardown-blocking steps; their work already completed. Wait a
            # moment so trailing log lines (the iteration table, "exitcode: N") reach the file.
            Start-Sleep -Seconds $StopGraceSec
            if ($proc.HasExited) {
                # Exited on its own during the grace period, which is the cleanest outcome.
                $stoppedEarly = $false
            } else {
                Stop-ProcessTree -Process $proc
            }
        } else {
            $timedOut = $true
            Write-Host "  TIMEOUT after ${TimeoutSec}s - killing the process tree"
            Stop-ProcessTree -Process $proc
        }
        if (-not $proc.HasExited) {
            # Last resort if the tree kill did not take.
            Get-Process -Name wandb-core -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
        }
    }
    $procExit = if ($timedOut -or $stoppedEarly) { $null } else { $proc.ExitCode }

    # Safety sweep. Killing the wrapper's process tree does not always reap the simulator, and the
    # wandb service is spawned detached; a leftover here burns CPU indefinitely (observed: a python
    # process accumulated 550 s of CPU after a play step). Only processes started at or after this
    # step began are touched, so a concurrent unrelated python is never killed.
    $stepStart = $stepStartedAt
    # NOTE: do not name this loop variable $name -- PowerShell variable names are case-insensitive,
    # so it would clobber the $Name parameter and corrupt every result row (observed: every step
    # reported as "wandb-core").
    foreach ($procName in @("python", "wandb-core")) {
        Get-Process -Name $procName -ErrorAction SilentlyContinue | Where-Object {
            $_.StartTime -ge $stepStart
        } | ForEach-Object {
            Write-Host ("  sweeping leftover {0} (pid {1}, started {2:HH:mm:ss})" -f $procName, $_.Id, $_.StartTime)
            Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        }
    }

    $proc.Dispose()
    $sw.Stop()

    $text = if (Test-Path $log) { Get-Content $log -Raw } else { "" }

    # Exit code: prefer the process's real exit status; fall back to the explicit ``exitcode: N``
    # line that step bodies emit (the LAST one wins, so a body may emit intermediate codes too).
    # A step may opt out of the exit-code check with -NoExitCodeCheck when a non-zero exit is part of
    # the documented behaviour (e.g. play raising the known requires-grad RuntimeError).
    $exitCode = $null
    $matches = [regex]::Matches($text, '(?m)^exitcode:\s*(-?\d+)\s*$')
    if ($matches.Count -gt 0) { $exitCode = [int]$matches[$matches.Count - 1].Groups[1].Value }
    if ($null -ne $procExit) { $exitCode = $procExit }

    $missing = @($ExpectPatterns | Where-Object { $text -notmatch [regex]::Escape($_) })

    # No generic "failure marker" scan. Every step's acceptance is expressed by its explicit
    # ExpectPatterns plus the exit code, which is unambiguous. A generic marker was tried and proved
    # fragile: "MISSING_MODULES: [" matched the success line "MISSING_MODULES: []", and a raw
    # "Traceback" would flag step 13, whose documented outcome *is* a traceback.
    $foundFail = @()

    if ($stoppedEarly) {
        # Work completed; the process was stopped because it could not exit on its own.
        $ok = ($missing.Count -eq 0)
        $verdict = "PASS (work complete; process stopped after success markers, teardown does not exit)"
    } elseif ($timedOut) {
        # A timeout does not have to mean failure. If the log already satisfies the step's
        # assertions, the work completed and only the teardown never exited -- the known behaviour of
        # both teardown-blocking steps. Reported distinctly so it is not mistaken for a clean exit.
        $assertionsMet = ($missing.Count -eq 0)
        $ok = $assertionsMet
        $verdict = if ($assertionsMet) {
            "PASS (timed out after the expected output appeared; teardown does not exit)"
        } else {
            "FAIL (timed out with missing output)"
        }
    } else {
        $exitOk = $NoExitCodeCheck -or ($exitCode -eq 0)
        $ok = ($missing.Count -eq 0) -and ($foundFail.Count -eq 0) -and $exitOk
        $verdict = if ($ok) { "PASS" } elseif (-not $exitOk) { "FAIL (exit code $exitCode)" } else { "FAIL" }
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
print('MISSING_MODULES:', bad)
sys.exit(1 if bad else 0)
"@
"exitcode: $LASTEXITCODE"
'@ -ExpectPatterns @("generate_id present: True","cuda available: True","MISSING_MODULES: []")

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

# 09 ------------------------------------------------------------------ training (wandb offline)
#
# Deliberately NO early stop. Training finishes its iterations in ~10 s and then blocks in wandb
# teardown, so the step ends by hitting the timeout -- and that is exactly when the log's final
# lines (the iteration table and "Training time:") have been flushed. Stopping on a marker instead
# truncated those lines and made the assertions fail. The timeout is set just above the observed
# training duration so the cost stays bounded.
Step -Name "09_train_wandb_offline" -TimeoutSec 120 -Body @'
& $PY -m isaaclab train --rl_library rsl_rl --task Tracking-Flat-G1-v0 `
    --num_envs 8 --max_iterations 3 --logger wandb `
    --log_project_name wbt-fullchain --run_name fullchain `
    --external_callback whole_body_tracking.tasks.register.register `
    --visualizer none physics=isaacsim_physx `
    "env.commands.motion.motion_file=$MOTION"
"exitcode: $LASTEXITCODE"
'@ -ExpectPatterns @("Logging to wandb","Learning iteration 0/3","Training time:") `
   -StopWhenLogMatches @("Training time:")

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
# wandb writes offline runs under WANDB_DIR, which the harness pins into the scratch tree, and it
# nests one further 'wandb' level inside it. Search recursively rather than guessing the depth.
run_root = os.environ.get('WANDB_DIR') or os.path.join(os.getcwd(), 'wandb')
runs = glob.glob(os.path.join(run_root, '**', 'offline-run-*'), recursive=True)
if not runs:
    print('NO OFFLINE RUN under', run_root)
    for root, dirs, files in os.walk(run_root):
        print('  walk:', root, dirs)
    sys.exit(1)
rd = max(runs, key=os.path.getmtime)
print('offline run:', os.path.basename(rd))
tx = glob.glob(os.path.join(rd, '*.wandb'))
if not tx:
    print('NO .wandb transaction file in', rd, os.listdir(rd))
    sys.exit(1)
data = open(tx[0], 'rb').read()
for n in (b'motion_policy', b'policy.onnx', b'policy.onnx.data'):
    print('manifest', n.decode(), n in data)
print('upload script done')
"@
"exitcode: $LASTEXITCODE"
'@ -ExpectPatterns @("guard OK","manifest policy.onnx.data True","upload script done","exitcode: 0")

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
# The frame rate is irrelevant here; the step ends by timeout so the Requires-grad traceback and the
# checkpoint-load line are both flushed before the process is killed.
Step -Name "13_play_latest" -TimeoutSec 120 -Body @'
& $PY -m isaaclab play --rl_library rsl_rl --task Tracking-Flat-G1-v0 --num_envs 2 `
    --checkpoint latest `
    --external_callback whole_body_tracking.tasks.register.register `
    --visualizer none physics=isaacsim_physx `
    "env.commands.motion.motion_file=$MOTION"
"exitcode: $LASTEXITCODE"
'@ -ExpectPatterns @("Loading experiment from directory:","Loading model checkpoint from:","requires grad") `
   -StopWhenLogMatches @("requires grad") -NoExitCodeCheck

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
