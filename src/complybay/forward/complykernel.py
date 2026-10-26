import numpy as np

from complybay.forward.ncomp_fortran import ncomp_fortran_batch


class ComplyKernel:
    def __init__(self, model, freq, depth, key, dz=None, dp=0.025):
        self.model = model
        self.freq = np.atleast_1d(np.asarray(freq, dtype=float))
        if self.freq.size != 1:
            raise ValueError("ComplyKernel takes a single frequency.")
        self.depth = depth
        self.key = key
        self.dz = dz
        self.dp = dp

        if self.dz is not None:
            thick, params = self._resample(self.model[:, 0], self.model[:, 1:], self.dz)
            self.model = np.column_stack((thick, params))

    def cal_kernel(self):
        key_map = {"thick": 0, "rho": 1, "vp": 2, "vs": 3}

        model = self.model
        freq = self.freq
        depth = self.depth
        dp = self.dp

        c0 = ncomp_fortran_batch(depth=depth, freqs=freq, model=model)
        kernel = np.zeros_like(model[:, 0], dtype=np.float64)
        d = model[:, 0].cumsum()

        idx = key_map.get(self.key, None)
        if idx is None:
            raise ValueError(f"Unknown key: {self.key}. Valid keys are: {list(key_map.keys())}")
        for i in range(model.shape[0]):
            val = model[i, idx]
            model[i, idx] = val * (1 + dp)
            c_plus = ncomp_fortran_batch(depth=depth, freqs=freq, model=model)

            model[i, idx] = val * (1 - dp)
            c_minus = ncomp_fortran_batch(depth=depth, freqs=freq, model=model)

            model[i, idx] = val  # Reset to original value

            kernel[i] = ((c_plus - c_minus) / (2 * dp * c0)).item()

        return d[:-1], kernel[:-1]

    def _resample(self, thick, v, dz):
        """
        Refer to https://github.com/keurfonluu/disba/blob/master/disba/_helpers.py
        """
        thick = np.asarray(thick)
        v = np.asarray(v)
        sizes = np.where(thick > dz, np.ceil(thick / dz).astype(int), 1)

        dzi = np.repeat(thick / sizes, sizes)

        if v.ndim == 1:
            vi = np.repeat(v, sizes)
        elif v.ndim == 2:
            vi = np.repeat(v, sizes, axis=0)
        else:
            raise ValueError("v must be 1D or 2D array")

        return dzi, vi


if __name__ == "__main__":
    depths = np.linspace(0, 5000, 101)
    vs = np.interp(depths, [0, 5000], [0.2, 3.0])

    f_vp = lambda vs: 1.16 * vs + 1.36
    f_rho = lambda vp: 1.74 * vp ** 0.25

    vp = f_vp(vs)
    rho = f_rho(vp)
    thick = depths[1:] - depths[:-1]
    model = np.column_stack((thick, rho[:-1], vp[:-1], vs[:-1]))

    freq = np.array([1/100])
    water_depth = 779

    ckernel = ComplyKernel(
        model=model,
        freq=freq,
        depth=water_depth,
        key='vs',
    )
    depth, kernel = ckernel.cal_kernel()
