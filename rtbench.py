#!/usr/bin/env python3
"""
Real-Time Benchmark Harness

This script loads a YAML suite definition and runs load generators and real-time
latency benchmarks (cyclictest) as specified. Results (JSON and histograms) are
saved in a structured results directory for later analysis.

Requirements:
  - Python 3.6+
  - PyYAML (pip install pyyaml)
  - stress-ng, cyclictest in PATH (may require sudo privileges)
"""
import argparse
import yaml
import subprocess
import time
import os
import shutil
import json
from pathlib import Path
from datetime import datetime


def load_yaml(path: Path) -> dict:
    with path.open('r') as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        yaml.dump(obj, f, sort_keys=False, indent=2)


def ensure_executable(cmd: list, exe_name: str) -> list:
    """
    Ensure that exe_name is the first element in cmd. Remove duplicates.
    """
    # Remove all existing occurrences
    filtered = [arg for arg in cmd if arg != exe_name]
    return [exe_name] + filtered


def build_cyclictest_cmd(base_cmd: list, json_out: Path, hist_out: Path) -> list:
    """
    Build the cyclictest command:
      - ensure 'cyclictest' is first
      - insert/replace --json <json_out>
      - insert/replace --histfile <hist_out>
    """
    cmd = list(base_cmd)

    # Ensure binary name
    cmd = ensure_executable(cmd, 'cyclictest')

    # Handle --json
    if '--json' in cmd:
        idx = cmd.index('--json')
        # Remove old argument if exists
        if idx + 1 < len(cmd) and not cmd[idx+1].startswith('--'):
            cmd.pop(idx+1)
    else:
        cmd.append('--json')
    cmd.insert(cmd.index('--json') + 1, str(json_out))

    # Handle histogram file
    if '--histfile' in cmd:
        idx = cmd.index('--histfile')
        cmd[idx+1] = str(hist_out)
    else:
        cmd.extend(['--histfile', str(hist_out)])

    return cmd


def run_subproc(cmd: list, cwd: Path = None, timeout: int = None) -> subprocess.CompletedProcess:
    print(f"    -> {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def run_benchmarks(suite_yaml: Path, results_dir: Path) -> None:
    config = load_yaml(suite_yaml)
    benchmarks = config.get('rtbench', [])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir.mkdir(parents=True, exist_ok=True)
    # snapshot the suite
    shutil.copy(suite_yaml, results_dir / f'suite_{timestamp}.yaml')

    for bench in benchmarks:
        name = bench.get('name', 'unnamed')
        runs = bench.get('global', {}).get('runs', 1)
        loadgen = bench['loadgen']['cmd']
        settle = bench['loadgen'].get('settle_sec', 0)
        base_ct = bench['lattest']['cmd']

        bench_dir = results_dir / name
        bench_dir.mkdir(exist_ok=True)
        print(f"[BENCH] {name}: {runs} runs")

        for run_i in range(1, runs + 1):
            run_dir = bench_dir / f"run{run_i:02d}"
            run_dir.mkdir(exist_ok=True)
            print(f"  Run {run_i}/{runs}")

            # Start load generator
            print(f"    Starting loadgen: {' '.join(loadgen)}")
            load_proc = subprocess.Popen(loadgen)

            # Wait settle
            if settle > 0:
                print(f"    Settling for {settle}s...")
                time.sleep(settle)

            # Prepare cyclictest invocation
            json_out = run_dir / 'cyclictest.json'
            hist_out = run_dir / 'histogram.txt'
            ct_cmd = build_cyclictest_cmd(base_ct, json_out, hist_out)

            # Execute cyclictest
            try:
                subprocess.run(ct_cmd, check=True)
                print(f"    cyclictest completed.")
            except subprocess.CalledProcessError as e:
                print(f"    [ERROR] cyclictest failed: {e}")

            # Teardown load generator
            if load_proc.poll() is None:
                print("    Stopping load generator")
                load_proc.terminate()
                try:
                    load_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    load_proc.kill()

            # Save run metadata
            meta = {
                'name': name,
                'run': run_i,
                'timestamp': datetime.now().isoformat(),
                'loadgen_cmd': loadgen,
                'cyclictest_cmd': ct_cmd,
                'json': str(json_out),
                'histogram': str(hist_out),
            }
            with (run_dir / 'metadata.json').open('w') as mf:
                json.dump(meta, mf, indent=2)

    print(f"Done. Results in {results_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Benchmark Harness")
    parser.add_argument('yaml', type=Path, help="Path to YAML suite definition")
    parser.add_argument('--results', type=Path, default=Path('results'), help="Directory to store results")
    parser.add_argument('--pretty-yaml', action='store_true', help="Write a prettified copy of the YAML")
    args = parser.parse_args()

    if args.pretty_yaml:
        out = args.yaml.with_suffix('.pretty.yaml')
        save_yaml(load_yaml(args.yaml), out)
        print(f"YAML rewritten: {out}")

    run_benchmarks(args.yaml, args.results)


if __name__ == '__main__':
    main()
