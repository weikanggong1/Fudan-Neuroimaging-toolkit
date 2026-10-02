"""Exact bounded GPU nearest-vertex search with the original KD-tree fallback.

A static uniform cell table is built once from mesh coordinates. Queries
inspect only adjacent cells; a geometric lower bound proves when every
unvisited cell is farther away. Unproven searches and floating-point distance
ties use cKDTree, preserving its original tie behavior. No dense Q x V
matrix is constructed. This helper changes candidate discovery, not the
radial face selection or interpolation arithmetic.
"""
import numpy as np
import torch


class ExactCellNearest:
    def __init__(self, vertices, device, *, chunk_size=4096):
        values = np.asarray(vertices, dtype=np.float64)
        self.device = torch.device(device)
        self.vertices = torch.as_tensor(values, device=self.device)
        self.chunk_size = chunk_size
        extent = np.ptp(values, axis=0)
        # About two spacings per cell on a sphere; the proof below does not
        # require a spherical, uniform or unfolded point cloud.
        radius = max(float(extent.max())/2, 1e-8)
        self.width = max(np.sqrt(4*np.pi*radius*radius/len(values))*2, 1e-8)
        self.origin = values.min(axis=0)-self.width
        cells = np.floor((values-self.origin)/self.width).astype(np.int64)
        shape = cells.max(axis=0)+2
        size = int(np.prod(shape))
        if size > 4_000_000:
            self.enabled = False
            return
        self.enabled = True
        self.shape = torch.as_tensor(shape, device=self.device)
        self.origin_t = torch.as_tensor(self.origin, device=self.device)
        linear = (cells[:, 0]*shape[1]+cells[:, 1])*shape[2]+cells[:, 2]
        order = np.argsort(linear, kind="stable")
        counts = np.bincount(linear, minlength=size)
        self.max_occupancy = int(counts.max())
        # Extreme clouds are cheap and more robust through the existing tree.
        if self.max_occupancy > 256:
            self.enabled = False
            return
        offsets = np.r_[0, counts.cumsum()]
        self.offsets = torch.as_tensor(offsets, device=self.device)
        self.order = torch.as_tensor(order, device=self.device)
        # Bound temporary gathered candidates to four million per chunk.
        self.chunk_size = min(self.chunk_size, max(1, 4_000_000//(27*self.max_occupancy)))
        self.slots = torch.arange(self.max_occupancy, device=self.device)
        self.neighbors = torch.cartesian_prod(*(torch.arange(-1, 2, device=self.device) for _ in range(3)))

    def query(self, points):
        """Return nearest IDs and a mask requiring original-tree resolution."""
        if not self.enabled:
            self.last_ties = torch.zeros(len(points), dtype=torch.bool, device=self.device)
            return (torch.zeros(len(points), dtype=torch.long, device=self.device),
                    torch.ones(len(points), dtype=torch.bool, device=self.device))
        answers = []; unresolved = []; ties = []
        for start in range(0, len(points), self.chunk_size):
            query = points[start:start+self.chunk_size]
            center = torch.floor((query-self.origin_t)/self.width).to(torch.long)
            cell = center[:, None, :]+self.neighbors[None]
            valid_cell = ((cell >= 0) & (cell < self.shape)).all(-1)
            clipped = torch.minimum(torch.maximum(cell, torch.zeros_like(cell)), self.shape-1)
            linear = (clipped[..., 0]*self.shape[1]+clipped[..., 1])*self.shape[2]+clipped[..., 2]
            beginning = self.offsets[linear]
            ending = self.offsets[linear+1]
            positions = beginning[..., None]+self.slots
            valid = valid_cell[..., None] & (positions < ending[..., None])
            safe = positions.clamp(max=len(self.order)-1)
            ids = self.order[safe].reshape(len(query), -1)
            valid = valid.reshape(len(query), -1)
            difference = self.vertices[ids]-query[:, None, :]
            square = difference*difference
            distance = (square[..., 0]+square[..., 1])+square[..., 2]
            distance = torch.where(valid, distance, torch.inf)
            best, index = distance.min(-1)
            nearest = ids.gather(1, index[:, None])[:, 0]
            low = self.origin_t+(center-1)*self.width
            high = self.origin_t+(center+2)*self.width
            bound = torch.minimum(query-low, high-query).min(-1).values
            # Strict proof with a conservative roundoff margin. An equal or
            # near-equal second distance is handed back to original cKDTree.
            margin = 16*torch.finfo(torch.float64).eps*torch.maximum(best, torch.ones_like(best))
            near_ties = ((distance-best[:, None]).abs() <= margin[:, None]).sum(-1) > 1
            proved = (bound > 0) & (best+margin < bound*bound) & torch.isfinite(best) & ~near_ties
            answers.append(nearest); unresolved.append(~proved); ties.append(near_ties)
        self.last_ties = torch.cat(ties)
        return torch.cat(answers), torch.cat(unresolved)
