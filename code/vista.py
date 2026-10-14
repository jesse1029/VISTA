"""Original VISTA: fixed-state PRISMA-only inference, with no target input."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

CLASSES = ("alfalfa", "almonds", "corn", "grapes", "pistachios")
STATE_KEYS = (
    "G_slope", "G_intercept", "decoder_mean", "decoder_scale",
    "decoder_coef", "decoder_intercept", "V_slope", "V_intercept", "V_scale",
)


def _require_finite(value, name):
    if not np.isfinite(value).all():
        raise ValueError(f"Non-finite {name}; measurements exceed numerical support")
    return value


@dataclass(frozen=True)
class Prediction:
    spectra: np.ndarray
    fallback: np.ndarray
    candidate: np.ndarray
    committed: np.ndarray
    annotation_index: np.ndarray


class FrozenVISTA:
    def __init__(self, state):
        self.state = {k: np.array(state[k], copy=True) for k in STATE_KEYS}
        self.wavelength_nm = np.array(state["wavelength_nm"], dtype=float, copy=True)
        s = self.state
        expected = {"G_slope": (59,), "G_intercept": (59,),
                    "decoder_mean": (59,), "decoder_scale": (59,),
                    "decoder_coef": (5, 59), "decoder_intercept": (5,),
                    "V_slope": (5, 59), "V_intercept": (5, 59)}
        for key, shape in expected.items():
            if s[key].shape != shape or not np.isfinite(s[key]).all():
                raise ValueError(f"Invalid frozen model field: {key}")
        if self.wavelength_nm.shape != (59,) or not np.isfinite(self.wavelength_nm).all() or not np.all(np.diff(self.wavelength_nm) > 0):
            raise ValueError("Expected the declared 59-band wavelength grid")
        if np.any(s["decoder_scale"] <= 0) or not 0 < float(s["V_scale"]) <= 1:
            raise ValueError("Invalid normalization or path scale")

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive["classes"].tolist()) != CLASSES:
                raise ValueError("Frozen classifier class order differs")
            return cls({k: archive[k] for k in (*STATE_KEYS, "wavelength_nm")})

    def probabilities(self, x):
        s = self.state
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            logits = ((x - s["decoder_mean"]) / s["decoder_scale"]) @ s["decoder_coef"].T
            logits = logits + s["decoder_intercept"]
            _require_finite(logits, "classifier logits")
            logits = logits - logits.max(axis=1, keepdims=True)
            _require_finite(logits, "centered classifier logits")
            p = np.exp(logits)
            return _require_finite(p / p.sum(axis=1, keepdims=True), "classifier probabilities")

    def predict(self, prisma_common59):
        x = np.asarray(prisma_common59, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 59 or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError("Expected a finite nonempty N-by-59 PRISMA array")
        s = self.state
        with np.errstate(over="ignore", invalid="ignore"):
            g = x * s["G_slope"] + s["G_intercept"]
        _require_finite(g, "fallback spectra")
        p = self.probabilities(g)
        full = g.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            for c in range(5):
                full += p[:, c:c+1] * (x * s["V_slope"][c] + s["V_intercept"][c])
            candidate = g + float(s["V_scale"]) * (full - g)
        _require_finite(full, "residual spectra")
        _require_finite(candidate, "candidate spectra")
        labels = np.argmax(p, axis=1)
        committed = np.argmax(self.probabilities(candidate), axis=1) == labels
        output = np.where(committed[:, None], candidate, g)
        if not np.array_equal(np.argmax(self.probabilities(output), axis=1), labels):
            raise ArithmeticError("Returned annotation constraint violated")
        return Prediction(output, g, candidate, committed, labels)

    def interpolate_prisma(self, native_prisma, native_wavelength_nm):
        raw = np.asarray(native_prisma, dtype=np.float64)
        waves = np.asarray(native_wavelength_nm, dtype=np.float64)
        if raw.ndim != 2 or waves.ndim != 1 or raw.shape[1] != len(waves):
            raise ValueError("Native wavelengths do not match the PRISMA array")
        if not np.isfinite(raw).all() or not np.isfinite(waves).all() or not np.all(np.diff(waves) > 0):
            raise ValueError("Expected finite measurements and strictly ascending wavelengths")
        query = self.wavelength_nm
        if len(waves) < 2 or waves[0] > query[0] or waves[-1] < query[-1]:
            raise ValueError("Cannot extrapolate outside native PRISMA wavelength support")
        hi = np.clip(np.searchsorted(waves, query, side="left"), 1, len(waves)-1)
        lo = hi - 1
        lo[waves[hi] == query] = hi[waves[hi] == query]
        denom = waves[hi] - waves[lo]
        fraction = np.divide(query-waves[lo], denom, out=np.zeros_like(query), where=denom != 0)
        with np.errstate(over="ignore", invalid="ignore"):
            interpolated = raw[:, lo] * (1-fraction) + raw[:, hi] * fraction
        return _require_finite(interpolated, "interpolated spectra")
