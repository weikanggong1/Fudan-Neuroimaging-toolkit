"""CPU-only Schur square-root prototype; never installs a production override."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import scipy.linalg
import torch
from fnit.synthmorph import models
import api_worker

original = models.matrix_sqrt


def cpu_schur(matrix):
    if matrix.device.type != 'cpu':
        return original(matrix)
    array = matrix.detach().numpy()
    flat = array.reshape(-1, 4, 4)
    values = np.stack([scipy.linalg.sqrtm(item) for item in flat])
    if np.iscomplexobj(values) or not np.isfinite(values).all():
        raise ValueError('prototype square root is not real and finite')
    return torch.from_numpy(values.reshape(array.shape)).to(matrix.dtype)


models.matrix_sqrt = cpu_schur
api_worker.main()
# The output arg is parsed by the existing worker; keep the complete CLI API.
import sys
output = Path(sys.argv[sys.argv.index('--output') + 1])
report = json.loads((output/'report.private.json').read_text())
report['prototype'] = {
    'scope': 'CPU-only scipy FP32 Schur matrix square root; original production model source unchanged; GPU branch explicitly unchanged',
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(output/'report.private.json').write_text(json.dumps(report, indent=2) + '\n')
# CLI comparison reader uses forward rather than the API save label transform.
os.link(output/'transform.nii.gz', output/'forward.nii.gz')
