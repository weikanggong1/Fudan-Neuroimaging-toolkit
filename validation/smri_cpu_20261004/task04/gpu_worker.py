"""Fixed-memory GPU regression wrapper for the same actual-image worker."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    import torch

    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(1.0, 20_000_000_000 / properties.total_memory), 0)
    torch.cuda.reset_peak_memory_stats(0)
    before = subprocess.run(['nvidia-smi', '--query-gpu=uuid,utilization.gpu,memory.used',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True)
    worker_path = Path(__file__).with_name('worker.py')
    specification = importlib.util.spec_from_file_location('fast_real_image_worker', worker_path)
    worker = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(worker)
    worker.main()
    output = Path(sys.argv[sys.argv.index('--output-dir') + 1])
    report_path = output / 'api_record.private.json'
    report = json.loads(report_path.read_text())
    report['gpu_memory'] = {
        'name': properties.name, 'total_bytes': properties.total_memory,
        'visible_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'allocator_budget_bytes': 20_000_000_000,
        'allocated_peak_bytes': torch.cuda.max_memory_allocated(0),
        'reserved_peak_bytes': torch.cuda.max_memory_reserved(0),
        'load_before_csv': before.stdout.strip(),
        'context_initialization_in_cold_process_clock': True,
    }
    report_path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
