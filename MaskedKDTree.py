from scipy.spatial import KDTree
import numpy as np
class MaskedKDTree(KDTree):
    def __init__(self, data, mask=None, **kwargs):
        super().__init__(data, **kwargs)
        self.mask = mask if mask is not None else np.ones(len(data), dtype=bool)

    def masked_distance(self, x, y):
        dx = x - y
        dx[self.mask[y]] = 0
        return np.sqrt((dx**2).sum())

    def query(self, x, k=1, **kwargs):
        distances, indices = super().query(x, k=k, distance_upper_bound=np.inf, **kwargs)
        for i, idx in enumerate(indices):
            mask = np.ones(len(self.data), dtype=bool)
            mask[idx] = False
            distances[i] = self.masked_distance(x[i], self.data[idx])
            for j in range(i + 1, k):
                min_dist_idx = np.argmin(distances[:i+1])
                if distances[min_dist_idx] > self.masked_distance(x[i], self.data[indices[j]]):
                    distances[min_dist_idx] = self.masked_distance(x[i], self.data[indices[j]])
                    indices[min_dist_idx] = indices[j]
            indices[:i+1] = indices[np.argsort(distances[:i+1])]
            distances[:i+1] = np.sort(distances[:i+1])
        return distances, indices
