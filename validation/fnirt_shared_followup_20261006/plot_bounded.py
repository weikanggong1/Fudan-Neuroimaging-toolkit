"""Plot only existing anonymous scalar records; no MRI/model/solver execution."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prior',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    current=json.loads((root/'reports/v3/arithmetic/summary.public.json').read_text())
    prior=json.loads(args.prior.read_text())['states'][1]['native_trace']
    assembly=json.loads((root/'reports/v2/assembly/summary.public.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.9),layout='constrained')
    iterations=[row['iteration'] for row in current['rows']]
    axes[0].semilogy(iterations,[row['relative_after'] for row in prior],label='Saved native',color='#245a9d',linewidth=2.3)
    axes[0].semilogy(iterations,[row['relative_after'] for row in current['rows']],label='Combined arithmetic',color='#ce7b23',linestyle='--',linewidth=1.2)
    axes[0].axhline(1e-3,color='#444',linewidth=.8,linestyle=':')
    axes[0].set(xlabel='PCG iteration',ylabel='Relative residual',title='Same canonical A/RHS: exact 69 rounds')
    axes[0].legend(frameon=False,fontsize=8);axes[0].grid(alpha=.18)
    keys=['rhs_current_lm_vs_native','rhs_current_fsl_order_vs_native','H_current_vs_native']
    axes[1].barh(['Current LM RHS','FSL-order RHS','Current H'],[assembly[key]['relative_l2'] for key in keys],color=['#245a9d','#ce7b23','#688b7e'])
    axes[1].invert_yaxis();axes[1].set_xscale('log')
    axes[1].set(xlabel='Relative L2 difference to saved native',title='Current second accepted point, fixed lambda')
    axes[1].grid(axis='x',alpha=.18)
    fig.suptitle('Bounded real shared-state diagnostics; no full registration',fontsize=11)
    fig.savefig(root/'reports/bounded_results.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
