"""Run run_capture.py on every capture in data/drive/manifest.tsv (date, setup, title, Drive id, bytes, start guess).

Each raw file is downloaded, processed and deleted again (the disk cannot hold all of them; the originals stay on
Drive). May 4 files already on disk are used in place. Finished captures (summary.json present) are skipped.
Usage: run_drive.py [WORKERS]
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
RAW = HERE / 'data/drive/raw'
URL = 'https://drive.usercontent.google.com/download?id={}&export=download&confirm=t'


def one(row):
    date, setup, title, fid, size, start = row
    out = HERE / 'data/drive' / date / setup / title.removesuffix('.sigmf-data')
    if (out / 'summary.json').exists():
        return f'skip {out}'
    local = next(HERE.glob(f'data/c*/{title}'), None)
    data = local or RAW / f'{setup}__{title}'
    if not local:
        for _ in range(3):
            subprocess.run(['curl', '-sL', '-o', str(data), URL.format(fid)], check=False)
            if data.exists() and data.stat().st_size == int(size):
                break
        else:
            data.unlink(missing_ok=True)
            return f'FAILED download {title}'
        data.with_suffix('.sigmf-meta').write_text(json.dumps({'global': {'core:sample_rate': 500000.0}}))
    r = subprocess.run([PY, str(HERE / 'run_capture.py'), str(data), str(HERE / f'tles/{date}.tle'), str(out), start],
                       capture_output=True, text=True, cwd=HERE)
    (out / 'run.log').write_text(r.stdout + r.stderr) if out.exists() else None
    if not local:
        data.unlink(missing_ok=True)
        data.with_suffix('.sigmf-meta').unlink(missing_ok=True)
    return f'{"done" if r.returncode == 0 else "ERROR"} {out}'


if __name__ == '__main__':
    RAW.mkdir(parents=True, exist_ok=True)
    rows = [l.split('\t') for l in (HERE / 'data/drive/manifest.tsv').read_text().splitlines() if l]
    with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
        for msg in ex.map(one, rows):
            print(msg, flush=True)
