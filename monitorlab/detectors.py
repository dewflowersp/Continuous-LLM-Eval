"""Online detectors for a metric that arrives one sample at a time.

This is the part of the tutorial that turns a score into a monitor. Every
detector here is incremental: it sees each sample once, keeps O(1) or
bounded-window state, and never looks at the future. That constraint is what
makes them deployable, and it is also what makes them interesting -- a
threshold you fit on the whole run is not a monitor, it is a post-mortem.

Four complementary jobs:

``EWMAControlChart``
    Where should the signal be right now? Produces the adaptive band.
``RobustZScore``
    Is this one sample an outlier? Median/MAD, so a single wild spike does not
    inflate the very spread used to judge it.
``PageHinkley``
    Has the mean shifted for good? Catches a regime change a fixed threshold
    would either miss or alarm on forever.
``CUSUM``
    Is the signal creeping? Accumulates small deviations that never individually
    look wrong.

All are deliberately short enough to read on a projector.

One design decision runs through the whole module: no detector threshold is
expressed in the metric's own units. Everything is in multiples of a running
estimate of that metric's own noise. The metrics here differ by three orders of
magnitude in scale and by a factor of three in noise -- grounding wobbles about
0.055 between steps, biased framing about 0.17 on the same [0, 1] scale, and
latency is measured in seconds that move from five to forty when you swap a 3B
model for a 70B. A threshold set that suits any one of those will false-alarm or
go deaf on the others, and it goes stale the moment the deployment changes.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Literal

Direction = Literal["higher_is_better", "lower_is_better"]


@dataclass
class Band:
    """The adaptive expectation for one sample."""

    center: float
    lower: float
    upper: float
    ready: bool  # False while still warming up


@dataclass
class Verdict:
    """What the detectors concluded about one sample."""

    value: float
    band: Band
    z: float = 0.0
    outlier: bool = False
    change_point: bool = False
    creep: bool = False
    severity: Literal["ok", "warn", "alarm"] = "ok"
    reasons: list[str] = field(default_factory=list)

    @property
    def fired(self) -> bool:
        return self.severity != "ok"


class RunningSpread:
    """Streaming estimate of how much a signal normally moves between steps.

    Mean absolute deviation from the running mean: O(1) state, no window, and
    far less distorted by one malformed generation than a variance would be.

    Its job is to give the change detectors a unit. ``update`` returns the
    sample's deviation from the mean *divided by* that spread, so a caller can
    say "accumulate anything beyond half a noise-width" and have it mean the
    same thing for grounding in [0, 1] and for latency in seconds.

    The floor has an absolute and a relative term, matching ``RobustZScore``:
    a long quiet stretch would otherwise drive the spread towards zero and turn
    the next ordinary wobble into an enormous deviation.
    """

    def __init__(self, floor_abs: float = 0.002, floor_rel: float = 0.02):
        self.floor_abs = floor_abs
        self.floor_rel = floor_rel
        self.mean = 0.0
        self.spread = 0.0
        self.n = 0

    @property
    def floor(self) -> float:
        return max(self.floor_abs, self.floor_rel * abs(self.mean))

    def update(self, x: float) -> float:
        """Admit ``x`` and return its deviation from the mean, in spread units."""
        self.n += 1
        if self.n == 1:
            self.mean = x
            return 0.0
        previous = self.mean
        self.mean += (x - self.mean) / self.n
        # Measured against the pre-update mean so a step change widens the
        # spread estimate instead of being quietly absorbed into it.
        self.spread += (abs(x - previous) - self.spread) / self.n
        return (x - self.mean) / max(self.spread, self.floor)

    def reset(self) -> None:
        self.mean = 0.0
        self.spread = 0.0
        self.n = 0


class EWMAControlChart:
    """Exponentially weighted moving average and spread.

    The band is ``mean +/- k * sigma`` where both are themselves EWMAs, so the
    threshold tracks the signal instead of being asserted up front. A model
    that is legitimately noisier simply gets a wider band.

    ``lam`` is the smoothing factor: higher adapts faster but is easier to fool
    by a slow drift, because the band follows the drift instead of flagging it.
    That trade-off is worth showing live by moving the slider.
    """

    def __init__(self, lam: float = 0.25, k: float = 3.0, warmup: int = 6):
        self.lam = lam
        self.k = k
        self.warmup = max(2, warmup)
        self.mean = 0.0
        self.var = 0.0
        self.n = 0

    def update(self, x: float) -> Band:
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.var = 0.0
        else:
            delta = x - self.mean
            self.mean += self.lam * delta
            # EWMA of squared deviation, measured against the pre-update mean so
            # a step change registers in the spread rather than being absorbed.
            self.var = (1 - self.lam) * (self.var + self.lam * delta * delta)

        sigma = math.sqrt(max(self.var, 0.0))
        ready = self.n >= self.warmup
        # A floor on sigma stops a pathologically quiet warm-up from producing a
        # zero-width band that then alarms on every subsequent sample.
        spread = self.k * max(sigma, 1e-3)
        return Band(self.mean, self.mean - spread, self.mean + spread, ready)


class RobustZScore:
    """Rolling median/MAD z-score.

    Uses MAD rather than standard deviation because these metrics are spiky: one
    malformed generation can double a classical sigma and mask everything after
    it. The 0.6745 factor rescales MAD to be comparable to a standard deviation
    for normal data, so the usual "z > 3" intuition still applies.

    MAD is floored before dividing. Without that, a short run of nearly
    identical samples drives MAD towards zero and every subsequent sample scores
    as a huge outlier -- which in testing produced z of +18 on an ordinary
    two-percent wobble. The floor encodes "differences below this are noise, not
    signal": an absolute term for metrics that live near zero, and a relative
    term so a metric on a larger scale (latency in seconds) gets a
    proportionate floor.
    """

    def __init__(
        self,
        window: int = 20,
        threshold: float = 3.5,
        min_history: int = 8,
        mad_floor_abs: float = 0.005,
        mad_floor_rel: float = 0.02,
    ):
        self.window = max(5, window)
        self.threshold = threshold
        self.min_history = max(5, min_history)
        self.mad_floor_abs = mad_floor_abs
        self.mad_floor_rel = mad_floor_rel
        self.values: Deque[float] = deque(maxlen=self.window)

    def update(self, x: float) -> tuple[float, bool]:
        # Score against history only, before admitting x, so a sample is never
        # used to normalise itself.
        history = list(self.values)
        self.values.append(x)
        if len(history) < self.min_history:
            return 0.0, False

        median = _median(history)
        mad = _median([abs(v - median) for v in history])
        floor = max(self.mad_floor_abs, self.mad_floor_rel * abs(median))
        z = 0.6745 * (x - median) / max(mad, floor)
        return z, abs(z) > self.threshold


class PageHinkley:
    """Page-Hinkley test for a persistent shift in the mean.

    Accumulates how far the signal has strayed from its running mean in one
    direction, ignoring anything smaller than ``delta`` as noise. When that
    accumulation exceeds ``lam`` the mean has moved and stayed moved. Two sums
    run at once, one looking for a rise and one for a drop, each compared
    against its own running extreme -- that "distance from the best it has been"
    construction is what makes the test fire on a sustained change rather than
    on a single bad sample.

    This is the detector that earns its keep when a model, prompt or corpus is
    swapped underneath the stream: a fixed threshold either never fires or fires
    forever, whereas this fires once, shortly after the change.

    ``delta`` and ``lam`` are in noise-widths, not metric units. ``lam = 3``
    means "three noise-widths of drift have piled up in one direction".
    """

    def __init__(
        self,
        delta: float = 0.5,
        lam: float = 3.0,
        direction: str = "both",
        min_n: int = 5,
    ):
        self.delta = delta
        self.lam = lam
        self.direction = direction
        # The spread estimate is worthless on two samples, and acting on it
        # produces an alarm before the monitor has seen normal behaviour.
        self.min_n = max(3, min_n)
        self.scale = RunningSpread()
        self.rise_sum = 0.0
        self.rise_min = 0.0
        self.drop_sum = 0.0
        self.drop_max = 0.0

    def update(self, x: float) -> bool:
        deviation = self.scale.update(x)
        if self.scale.n < self.min_n:
            return False

        self.rise_sum += deviation - self.delta
        self.rise_min = min(self.rise_min, self.rise_sum)
        self.drop_sum += deviation + self.delta
        self.drop_max = max(self.drop_max, self.drop_sum)

        rose = (self.rise_sum - self.rise_min) > self.lam
        dropped = (self.drop_max - self.drop_sum) > self.lam

        fired = (
            (self.direction == "down" and dropped)
            or (self.direction == "up" and rose)
            or (self.direction == "both" and (dropped or rose))
        )
        if fired:
            self.reset()
        return fired

    def reset(self) -> None:
        """Re-baseline after firing, so one change yields one alarm.

        The spread estimate is dropped too: after a regime change the signal's
        noise is usually different, and the detector should relearn it rather
        than judge the new regime by the old one's standards.
        """
        self.scale.reset()
        self.rise_sum = self.rise_min = 0.0
        self.drop_sum = self.drop_max = 0.0


class CUSUM:
    """Cumulative sum, for drift too gradual to trip the other tests.

    Each sample contributes its deviation from the running mean, minus a slack
    term; the sum is floored at zero so noise cannot bank credit. Sustained
    small deviations in one direction eventually clear ``threshold``.

    Like Page-Hinkley, both parameters are in noise-widths. The difference
    between the two is patience: Page-Hinkley wants a shift, CUSUM will also
    catch a slow slide where no individual step looks wrong.
    """

    def __init__(
        self,
        slack: float = 0.5,
        threshold: float = 4.0,
        direction: str = "both",
        min_n: int = 5,
    ):
        self.slack = slack
        self.threshold = threshold
        self.direction = direction
        self.min_n = max(3, min_n)
        self.scale = RunningSpread()
        self.pos = 0.0
        self.neg = 0.0

    def update(self, x: float) -> bool:
        deviation = self.scale.update(x)
        if self.scale.n < self.min_n:
            return False

        self.pos = max(0.0, self.pos + deviation - self.slack)
        self.neg = max(0.0, self.neg - deviation - self.slack)

        fired = (
            (self.direction == "up" and self.pos > self.threshold)
            or (self.direction == "down" and self.neg > self.threshold)
            or (self.direction == "both" and (self.pos > self.threshold or self.neg > self.threshold))
        )
        if fired:
            self.pos = self.neg = 0.0
        return fired


class MetricMonitor:
    """The four detectors wired together for one metric.

    Only excursions in the bad direction are escalated: for grounding, dropping
    out of the band is an alarm while rising out of it is good news. That applies
    to all four detectors, including the z-score -- swapping in a model that
    scores *better* is reported but not escalated. That is what ``direction``
    encodes, and it is the difference between a monitor people trust and one they
    mute. It is also why a latency *improvement* during a quality collapse stays
    green, which is worth pointing at: the operational dashboard would have
    looked fine.

    The ``ph_*`` and ``cusum_*`` parameters are in noise-widths, so this one set
    of numbers serves every metric. ``scripts/calibrate.py`` sweeps them against
    recorded runs and reports false alarms and detection delay.
    """

    def __init__(
        self,
        metric: str,
        direction: Direction = "higher_is_better",
        lam: float = 0.25,
        k: float = 3.0,
        warmup: int = 6,
        z_window: int = 20,
        z_threshold: float = 4.0,
        ph_delta: float = 0.75,
        ph_lambda: float = 3.0,
        cusum_slack: float = 0.5,
        cusum_threshold: float = 4.0,
        mad_floor_abs: float = 0.005,
        mad_floor_rel: float = 0.02,
    ):
        self.metric = metric
        self.direction = direction
        bad = "down" if direction == "higher_is_better" else "up"
        self.chart = EWMAControlChart(lam=lam, k=k, warmup=warmup)
        self.zscore = RobustZScore(
            window=z_window,
            threshold=z_threshold,
            mad_floor_abs=mad_floor_abs,
            mad_floor_rel=mad_floor_rel,
        )
        # Change and creep detectors watch only the harmful direction.
        self.page_hinkley = PageHinkley(delta=ph_delta, lam=ph_lambda, direction=bad)
        self.cusum = CUSUM(slack=cusum_slack, threshold=cusum_threshold, direction=bad)

    def update(self, value: float) -> Verdict:
        band = self.chart.update(value)
        z, is_outlier = self.zscore.update(value)
        changed = self.page_hinkley.update(value)
        creeping = self.cusum.update(value)

        harmful = z < 0 if self.direction == "higher_is_better" else z > 0
        worse_than_band = (
            value < band.lower if self.direction == "higher_is_better" else value > band.upper
        )
        # A band excursion only counts once the band means something.
        breached = band.ready and worse_than_band

        reasons: list[str] = []
        if breached:
            edge = band.lower if self.direction == "higher_is_better" else band.upper
            reasons.append(f"outside adaptive band ({value:.3f} vs {edge:.3f})")
        if is_outlier:
            # An outlier in the good direction is still reported, because
            # something clearly changed and the room should see that the monitor
            # noticed. It just does not raise the severity: swapping in a model
            # that grounds better should not page anyone.
            reasons.append(f"robust z={z:+.1f}" if harmful
                           else f"robust z={z:+.1f} (improvement, not escalated)")
        if changed:
            reasons.append("Page-Hinkley: sustained shift in the mean")
        if creeping:
            reasons.append("CUSUM: gradual drift")

        bad_outlier = is_outlier and harmful
        if changed or (breached and bad_outlier):
            severity = "alarm"
        elif breached or creeping or bad_outlier:
            severity = "warn"
        else:
            severity = "ok"

        return Verdict(
            value=value,
            band=band,
            z=z,
            outlier=is_outlier,
            change_point=changed,
            creep=creeping,
            severity=severity,
            reasons=reasons,
        )


def default_params() -> dict[str, float]:
    """``MetricMonitor``'s own keyword defaults.

    The UI reads its slider defaults from here rather than repeating the numbers,
    because the two drifted apart once already: the detectors were retuned into
    noise-width units while the sliders kept offering the old metric-unit ranges,
    which quietly fed nonsense thresholds to every monitor.
    """
    import inspect

    return {
        name: parameter.default
        for name, parameter in inspect.signature(MetricMonitor.__init__).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])
