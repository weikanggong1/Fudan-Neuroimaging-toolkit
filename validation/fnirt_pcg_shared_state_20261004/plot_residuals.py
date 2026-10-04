"""Plot anonymous real shared-system residuals; no imaging inputs required."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

reports = Path(__file__).parent / 'reports'
figure, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
labels = {'native': 'Native CG', 'fnit_dense': 'FNIT dense',
          'fnit_column_order': 'FNIT column order',
          'control_column_division': 'Column + division control'}
for solve, axis in zip([2, 3], axes):
    with (reports / f'solve{solve}_residuals.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    for key, label in labels.items():
        selected = [row for row in rows if row[key]]
        x = [int(row['iteration']) for row in selected]
        y = [float(row[key]) for row in selected]
        line, = axis.semilogy(x, y, label=f'{label} ({len(x)})', linewidth=1.3, alpha=.85)
        axis.plot(x[-1], y[-1], 'o', color=line.get_color(), markersize=4)
    axis.axhline(1e-3, color='black', linestyle='--', linewidth=.9, label='Fixed tolerance 1e-3')
    axis.set_title(f'Shared continuation solve {solve}')
    axis.set_xlabel('PCG iteration')
    axis.grid(True, which='major', alpha=.2)
axes[0].set_ylabel('Relative unpreconditioned residual')
axes[1].legend(fontsize=8, loc='upper right')
figure.suptitle('Same real A, RHS, diagonal and zero initial state', fontsize=11)
figure.tight_layout()
figure.savefig(reports / 'residuals.png', dpi=160)
