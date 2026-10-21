"""Redraw the plots of a run from its logs (no GPU, no torch needed).

    python scripts/plot_run.py runs/seed1
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from plotting import plot_search, plot_retrain


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    run = sys.argv[1]
    s = os.path.join(run, 'search')
    if os.path.exists(os.path.join(s, 'epochs.jsonl')):
        rows = read_jsonl(os.path.join(s, 'epochs.jsonl'))
        classes = None
        if os.path.exists(os.path.join(s, 'meta.json')):
            classes = json.load(open(os.path.join(s, 'meta.json')))['classes']
        plot_search(rows, os.path.join(s, 'search_curves.png'), classes=classes)
        print('wrote', os.path.join(s, 'search_curves.png'))
    r = os.path.join(run, 'retrain')
    if os.path.exists(os.path.join(r, 'epochs.jsonl')):
        plot_retrain(read_jsonl(os.path.join(r, 'epochs.jsonl')), os.path.join(r, 'retrain_curves.png'))
        print('wrote', os.path.join(r, 'retrain_curves.png'))


if __name__ == '__main__':
    main()
