"""Publish genuine Phase 4 checks; call only after final tests and GPU probes."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from cls_balance.artifacts import environment, file_hash, object_hash, seed_everything, write_json
from cls_balance.report_flat import read_json
from cls_balance.train_independent import unique_id

parser = argparse.ArgumentParser()
parser.add_argument('--report', type=Path, required=True)
parser.add_argument('--residency', type=Path, required=True)
parser.add_argument('--output', type=Path, help='Reserved immutable validation directory; must not exist')
args = parser.parse_args()
repo = Path.cwd()
output = args.output.resolve() if args.output else repo / 'results/stage4' / unique_id('validation')
output.mkdir(parents=True, exist_ok=False)
for name, source in [('pytest.txt', '.tmp-phase4-final-pytest.log'), ('junit.xml', '.tmp-phase4-final-junit.xml'),
                     ('training-console.txt', '.tmp-phase4-training-v2.log')]:
    # Normalize captured Windows console newlines before computing publication hashes.
    (output / name).write_text((repo / source).read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
suites = ET.parse(output / 'junit.xml').getroot().findall('testsuite')
counts = {key: sum(int(s.get(key, 0)) for s in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
if counts['tests'] < 213 or any(counts[k] for k in ('failures', 'errors', 'skipped')):
    raise ValueError('Final validation is not complete and passing')
for name, command in [('freeze-check.txt', ['.venv/Scripts/python.exe', 'scripts/freeze_tests.py', '--check']),
                      ('pip-check-cpu.txt', ['.venv/Scripts/python.exe', '-m', 'pip', 'check']),
                      ('pip-check-cuda.txt', ['.venv-cuda/Scripts/python.exe', '-m', 'pip', 'check'])]:
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    (output / name).write_text(result.stdout + result.stderr, encoding='utf-8', newline='\n')
publication = read_json(args.report / 'publication.json')
for name, digest in publication['artifacts_sha256'].items():
    if file_hash(args.report / name) != digest:
        raise ValueError('Publication hash mismatch')
residency = read_json(args.residency / 'summary.json')
for name, digest in residency['artifacts_sha256'].items():
    if file_hash(args.residency / name) != digest:
        raise ValueError('Residency probe hash mismatch')
if not residency['prediction_and_path_equality']:
    raise ValueError('Residency changed predictions')
shutil.copytree(args.residency, output / 'residency')
(output / 'residency-initialization-failure.txt').write_text((repo / '.tmp-phase4-residency-initialization-failure.log').read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cls_balance.report_flat import save_svg
with (output / 'residency/probes.csv').open(encoding='utf-8', newline='') as stream:
    probes = list(csv.DictReader(stream))
policies = residency['policies']
groups = [[p for p in probes if p['policy'] == policy] for policy in policies]
plt.rcParams.update({'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].boxplot([[float(p['unwarmed_model_path_ms']) for p in group] for group in groups], tick_labels=policies)
axes[0].set(yscale='log', ylabel='Un-warmed ready-tensor model path (ms; log scale)', title='30 validation images per policy')
axes[1].bar(policies, [float(group[0]['setup_seconds']) for group in groups], color='#126782')
axes[1].set(ylabel='Setup wall seconds', title='Construction / preload / initial device transfer')
axes[2].bar(policies, [max(float(p['peak_gpu_allocated_bytes']) for p in group) / 2**20 for group in groups], color='#527d39')
axes[2].set(ylabel='Peak allocated GPU memory (MiB)', title='All stored CNN artifacts counted in every policy')
for ax in axes:
    ax.tick_params(axis='x', rotation=20)
fig.suptitle('Validation residency diagnostics · synchronized FP32 · not full-pipeline test p95/FPS', fontsize=13)
fig.tight_layout(rect=(0, 0, 1, .92))
save_svg(fig, output / 'residency.svg')
fig.savefig(repo / '.tmp/residency-preview.png', dpi=120)
plt.close(fig)
old = repo / 'runs/independent/ind-s42-20261026T125510Z-3fa0423c'
old_epochs = list(old.glob('nodes/*/*/epoch-*.json'))
new_rows = []
for line in (repo / '.tmp-phase4-training-v2.log').read_text(encoding='utf-8').splitlines():
    try:
        row = json.loads(line)
        if row.get('seed') == 42 and row.get('node') == 'manufacturer' and row.get('epoch') == 1:
            new_rows.append(row)
    except ValueError:
        pass
old_row = read_json(old_epochs[0])
fields = [f'{split}_{metric}' for split in ('train', 'val') for metric in ('loss', 'macro_recall', 'top1', 'images', 'optimizer_steps')]
if any(old_row[k] != new_rows[0][k] for k in fields):
    raise ValueError('Loader restart changed initial scientific observations')
interruption = {'run_id': old.name, 'status': 'interrupted; no complete checkpoint or cascade',
                'partial_epoch_measurements': 'not retained; incomplete attempt excluded from successful-run resource totals',
                'reason': 'Restarted with parallel decoding for large branches; serial image loading was costly',
                'completed_epochs': len(old_epochs), 'epoch1_metrics_identical_after_restart': True,
                'serial_epoch_seconds': old_row['train_wall_seconds'] + old_row['val_wall_seconds'],
                'parallel_epoch_seconds': new_rows[0]['train_wall_seconds'] + new_rows[0]['val_wall_seconds'],
                'compared_fields': fields}
write_json(output / 'interrupted-attempt.json', interruption)
paused_run = repo / 'runs/independent/ind-s44-20261026T153237Z-2b462c75'
paused_attempts = []
for contract_path in sorted(paused_run.glob('nodes/*/*/contract.json')):
    attempt = contract_path.parent
    if (attempt / 'summary.json').exists():
        continue
    if attempt.parent.name != 'manufacturer':
        continue
    epochs = [read_json(p) for p in sorted(attempt.glob('epoch-*.json'))]
    paused_attempts.append({'attempt': attempt.relative_to(paused_run).as_posix(),
        'status': 'incomplete; no exported selected node checkpoint',
        'completed_epochs': len(epochs),
        'completed_epoch_training_exposures': sum(r['train_images'] for r in epochs),
        'completed_epoch_wall_seconds': sum(r['train_wall_seconds'] + r['val_wall_seconds'] for r in epochs),
        'partial_epoch_measurements': 'not retained; excluded from successful-run resource totals'})
if not paused_attempts or sum(a['completed_epochs'] for a in paused_attempts) != 7:
    raise ValueError('Paused seed-44 attempt inventory differs from the recorded seven epochs')
completed_router = next(paused_run.glob('nodes/manufacturer/*/summary.json')).parent
restart_matches = []
for attempt in paused_attempts:
    for old_epoch in sorted((paused_run / attempt['attempt']).glob('epoch-*.json')):
        previous, restarted = read_json(old_epoch), read_json(completed_router / old_epoch.name)
        restart_matches.append(all(previous[k] == restarted[k] for k in fields))
write_json(output / 'user-pause-attempt.json', {'run_id': paused_run.name,
    'reason': 'User requested a pause; completed seeds 42/43 were retained; seed-44 router restarted on explicit continue',
    'initial_run_git_sha': read_json(paused_run / 'environment.json')['git_sha'],
    'resumed_evaluation_git_sha': read_json(args.report / 'seed-44/IND-2-predicted/metadata.json')['environment']['git_sha'],
    'completed_epoch_metrics_exactly_reproduced': all(restart_matches),
    'exactly_matching_epochs': sum(restart_matches),
    'attempts': paused_attempts, 'successful_run_resources_exclude_these_attempts': True})
sleep_attempts = []
failed_launch_attempts = []
for contract_path in sorted(paused_run.glob('nodes/variant-manufacturer-1/*/contract.json')):
    attempt = contract_path.parent
    if (attempt / 'summary.json').exists():
        continue
    epochs = [read_json(p) for p in sorted(attempt.glob('epoch-*.json'))]
    if not epochs:
        failed_launch_attempts.append({'attempt': attempt.relative_to(paused_run).as_posix(),
            'completed_epochs': 0, 'partial_epoch_measurements': 'unmeasured; no exported selected checkpoint'})
        continue
    sleep_attempts.append({'attempt': attempt.relative_to(paused_run).as_posix(),
        'status': 'discarded before checkpoint export; timing contaminated by system sleep',
        'completed_epochs': len(epochs),
        'completed_epoch_training_exposures': sum(r['train_images'] for r in epochs),
        'observed_epoch_wall_seconds_including_sleep': sum(r['train_wall_seconds'] + r['val_wall_seconds'] for r in epochs),
        'maximum_validation_epoch_wall_seconds': max(r['val_wall_seconds'] for r in epochs),
        'active_gpu_time': None, 'partial_epoch_measurements': 'unmeasured'})
if len(sleep_attempts) != 1 or sleep_attempts[0]['completed_epochs'] != 20:
    raise ValueError('System-sleep discarded attempt differs from the recorded 20 epochs')
write_json(output / 'system-sleep-attempt.json', {'run_id': paused_run.name,
    'reason': 'Windows sleep/resume stalled Airbus validation; only the affected incomplete node was restarted',
    'power_event': read_json(repo / '.tmp/sleep-event.json'), 'attempts': sleep_attempts,
    'successful_run_resources_exclude_these_attempts': True,
    'mitigation': 'Temporary process-lifetime ES_SYSTEM_REQUIRED request; no power-plan or clock changes',
    'sleep_guard_sha256': file_hash(repo / '.tmp/train_awake.py')})
shutil.copyfile(repo / '.tmp/train_awake.py', output / 'train_awake.py')
if len(failed_launch_attempts) != 2:
    raise ValueError('Unexpected failed Windows launcher attempt inventory')
write_json(output / 'failed-launcher-attempts.json', {'run_id': paused_run.name,
    'reason': 'Initial auxiliary sleep-guard launcher lacked Windows multiprocessing main guard; corrected before retry',
    'attempts': failed_launch_attempts, 'successful_run_resources_exclude_these_attempts': True,
    'smoke_check': 'spawn pool returned exact requested value; exit 0; actual Torch DataLoader retried normally',
    'failed_source_sha256': file_hash(repo / '.tmp/train_awake.failed-import.py'),
    'corrected_source_sha256': file_hash(repo / '.tmp/train_awake.py')})
shutil.copyfile(repo / '.tmp/train_awake.failed-import.py', output / 'failed-launcher-source.txt')
(output / 'awake-smoke.txt').write_text((repo / '.tmp/awake-smoke.txt').read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
seed_everything(42)
write_json(output / 'environment.json', environment(repo))
sources = [*repo.glob('src/**/*.py'), *repo.glob('checks/*.py'), *repo.glob('configs/*.yaml'),
           *repo.glob('docs/*.md'), repo / 'README.md', repo / '.gitignore', repo / '.gitattributes',
           repo / '.github/workflows/phase4.yml']
write_json(output / 'source_hashes.json', {p.relative_to(repo).as_posix(): file_hash(p) for p in sorted(sources)})
shutil.copyfile(__file__, output / 'publish_validation.py')
validation = {'status': 'verified', **counts, 'frozen_baseline_tests': 172,
              'source_run_ids': publication['source_run_ids'], 'training_report': args.report.as_posix(),
              'training_publication_sha256': file_hash(args.report / 'publication.json'),
              'residency_summary': residency, 'test_inference_performed': False,
              'dataset_images_weights_embeddings_published': False}
write_json(output / 'validation.json', validation)
text = f'''# Phase 4 validation evidence

{counts['tests']} tests passed with no failures, errors or skips. The original
172-test suite and runner inputs pass their frozen SHA256 audit. Both pinned CPU
and CUDA environments pass `pip check`.

The Phase 4 checks cover real EfficientNet export/reload, independent parameter
storage, taxonomy-to-local target mapping, singleton exclusion, two- and
three-stage optimizer/bundle/evaluation orchestration on generated fixtures,
hard/oracle routes, residency equality, verified node reuse, tamper rejection
and publication exclusion of training-ID contracts. Generated fixtures are not
aircraft measurements.

The [measured training report](../{args.report.name}/REPORT.md) contains genuine
full validation predictions for seeds 42/43/44. [Residency probes](residency/summary.json)
compare all-resident, preloaded-CPU and cold-disk paths on the same 30 validation
images. These un-warmed ready-tensor probes are not test p95/FPS measurements.
The first probe failed before recording measurements because CUDA peak-stat
reset preceded allocator initialization. A released sentinel allocation now
bootstraps the allocator before all policy timers; CUDA runtime startup is
excluded. The corrected probe passed and its raw failed-start log is retained.

![Validation residency diagnostics](residency.svg)

The [interrupted attempt](interrupted-attempt.json) completed one epoch before
the decoding-policy restart. No partial checkpoint is published as a result.
Its epoch-one accuracy/loss observations exactly match the restarted run.
Any partial epoch after that completed epoch is unmeasured; successful-run
resource totals exclude the abandoned attempt.
The [user-requested pause](user-pause-attempt.json) interrupted seed 44 after
seven complete router epochs. Those metrics were retained, while the incomplete
node restarted in a new attempt on explicit continuation. Its discarded-epoch
costs and any unmeasured partial epoch are excluded from successful-run totals.
Exact scientific metric reproduction: {sum(restart_matches)} of {len(restart_matches)}
completed pre-pause epochs (wall-clock timings excluded).
The [system-sleep attempt](system-sleep-attempt.json) retained 20 discarded Airbus
epochs, including a validation epoch spanning the 6.2-hour sleep. This elapsed
time is not active GPU time. Only the affected incomplete node restarted; all
38 previously completed CNNs were preserved. Successful-run resources exclude
that entire attempt. A temporary process-lifetime idle-sleep request protected
the retry without altering the OS power plan, display settings or clock policy.
The first auxiliary launcher failed Windows multiprocessing startup before any
new epoch completed. Its [two empty attempts](failed-launcher-attempts.json) were
retained; a main import guard corrected the launcher and a separate spawn smoke
check passed before the successful retry. The core trainer source stayed unchanged.
Source/configuration/environment snapshots and console output are included;
dataset images, training-ID lists, embeddings and weights remain local.

## Exact checks

```powershell
& .\\.venv\\Scripts\\python.exe scripts/freeze_tests.py --check
& .\\.venv\\Scripts\\python.exe -m pytest -q tests checks --junitxml=.tmp-phase4-final-junit.xml
& .\\.venv\\Scripts\\python.exe -m pip check
& .\\.venv-cuda\\Scripts\\python.exe -m pip check
```

IND-3 real-data training is gated after the primary two-stage comparison.
Test inference and full-pipeline speed benchmarks remain deferred. Training
wall time includes ordinary OS/background load; CPU checks overlapped early
router epochs. No equal-compute or deployment-speed claim is made.
'''
(output / 'REPORT.md').write_text(text, encoding='utf-8', newline='\n')
write_json(output / 'publication.json', {'status': 'complete', 'scope': 'Phase 4 validation and residency evidence',
           'artifacts_sha256': {p.relative_to(output).as_posix(): file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()}})
print(output)
