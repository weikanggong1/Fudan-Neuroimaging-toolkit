"""Plot saved real-stage scalar diagnostics; no image or private mesh loading."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rows=json.loads(args.summary.read_text())['rows']
    steps=[r['step'] for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.0),layout='constrained')
    axes[0].semilogy(steps,[r['actual_Double_coordinate_difference']['max_abs'] for r in rows],color='#286c91',marker='.',ms=5)
    axes[0].set_ylabel('Maximum coordinate difference (voxel)')
    axes[0].set_title('Actual saved Double points vs native')
    axes[1].semilogy(steps,[r['actual_CPU_alpha'] for r in rows],label='CPU actual',color='#286c91',marker='.',ms=5)
    axes[1].semilogy(steps,[r['native_alpha_inferred_from_saved_Deformation_and_source_direction'] for r in rows],label='Native inferred',color='#b76a29',marker='.',ms=4)
    axes[1].set_ylabel('Accepted line-search alpha');axes[1].set_title('First alpha fork: step5');axes[1].legend(fontsize=9)
    for ax in axes:
        ax.set_xlabel('Saved accepted step');ax.grid(alpha=.25);ax.axvline(5,color='#777777',ls=':',lw=1)
    fig.suptitle('Fixed real T1-derived label stage; no complete recipe')
    fig.savefig(args.output,dpi=150);plt.close(fig)


if __name__=='__main__':main()
