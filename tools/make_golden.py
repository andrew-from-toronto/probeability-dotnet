"""Inspect Probe-ability's compiled model and emit golden vectors for a C# port.

Two sets:
  model_cases   : raw 17-float feature vector -> score() minutes (isolates the tree evaluator)
  pipeline_cases: reading stream (ts, internal C, ambient C) + target + cook_name + start_temp
                  -> features (17 floats) and MLPredictor.predict() minutes (tests feature building)
"""
import json, math, os, random, struct, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
# update.py points these at a fresh checkout; run by hand they default to ./repo.
PKG = os.environ.get("PROBEABILITY_PKG") or os.path.join(HERE, "repo", "custom_components", "probe_ability")
sys.path.insert(0, PKG)

import ml_model_code as m          # noqa: E402
import ml_predictor as mp          # noqa: E402

# ---- structure stats ------------------------------------------------------
blob = m._DATA
def stats(offsets):
    nodes = sum(c for _, c in offsets)
    feats = collections.Counter()
    depth_max = 0
    for start, count in offsets:
        for n in range(count):
            f, th, l, r, v = struct.unpack_from(m._FMT, blob, start + n * m._NS)
            if l != -1:
                feats[f] += 1
    return len(offsets), nodes, feats
near = stats(m._OFFSETS)
far = stats(m._OFFSETS_FAR)
print("blob bytes", len(blob), "node records", len(blob) // m._NS)
print("near trees/nodes", near[0], near[1]); print("far trees/nodes", far[0], far[1])
print("split usage near", sorted(near[2].items())); print("split usage far", sorted(far[2].items()))
last_end = max(s + c * m._NS for s, c in m._OFFSETS_FAR)
print("far ends at", last_end)

# sanity: leaf encoding
f, th, l, r, v = struct.unpack_from(m._FMT, blob, 0 + 30 * m._NS)
print("sample record", f, th, l, r, v)

def f64(x):
    return x  # json dumps repr -> round-trips exactly

# ---- model cases ----------------------------------------------------------
ORDER = mp._FEATURE_ORDER
rng = random.Random(20260927)
model_cases = []
def add_model(name, x):
    model_cases.append({"name": name, "x": [float(v) for v in x], "minutes": m.score(x)})

meats = {
    "Beef Brisket Fall Apart": 96.0, "Pork Shoulder Pulled": 96.0, "Beef Rib Eye Medium Rare": 54.0,
    "Poultry Whole Bird Well Done": 82.0, "Lamb Leg Medium": 63.0, "Other Fish / Salmon Medium": 60.0,
    "Custom": 70.0, "Poultry Chicken Breast Medium": 74.0,
}
for cook, target in meats.items():
    enc = mp.resolve_meat(cook, target)
    for trem in (40.0, 20.0, 15.0, 12.0, 9.5, 6.0, 4.0, 2.0, 0.7):
        cur = target - trem
        amb = rng.choice([107.0, 120.0, 150.0, 180.0, 220.0])
        rate_i = max(0.02, rng.uniform(0.05, 1.5) * (trem / 40 + 0.2))
        rate_r = rate_i * rng.uniform(0.4, 1.2)
        stall = 1.0 if (abs(rate_r) < 0.2 and 40 <= cur <= 80) else 0.0
        x = [cur, amb, rate_i, rate_r, rate_r / rate_i if abs(rate_i) > 0.01 else 1.0,
             rng.uniform(2, 8), trem, rng.uniform(15, 600), amb + rng.uniform(-3, 3),
             rng.uniform(0, 4), stall, amb - cur, *[float(e) for e in enc]]
        add_model(f"{cook} trem={trem}", x)
# explicit stall + edge cases
bris = [float(e) for e in mp.resolve_meat("Beef Brisket Fall Apart", 96)]
add_model("brisket stall 68C", [68, 115, 0.3, 0.05, 0.1667, 4, 28, 360, 114, 1.5, 1, 47, *bris])
add_model("brisket no-stall 68C", [68, 115, 0.3, 0.25, 0.8333, 4, 28, 360, 114, 1.5, 0, 47, *bris])
add_model("negative T_remaining (overshoot)", [75, 180, 0.8, 0.6, 0.75, 5, -1, 60, 180, 0.5, 0, 105,
                                               *[float(e) for e in mp.resolve_meat("Poultry Chicken Breast Medium", 74)]])
add_model("decel default 1.0 (tiny initial rate)", [20, 120, 0.005, 0.005, 1.0, 5, 50, 12, 120, 0, 0, 100,
                                                    *[float(e) for e in mp.resolve_meat("Custom")]])
add_model("zero ambient std, exact T_FAR", [45, 150, 0.9, 0.8, 0.8889, 5, 15.0, 45, 150, 0, 0, 105, *bris])
add_model("exact T_NEAR", [92, 150, 0.2, 0.18, 0.9, 5, 4.0, 400, 150, 0, 0, 58, *bris])

# ---- pipeline cases -------------------------------------------------------
def simulate(ambient, start, target, k, stall_at=None, stall_min=0, minutes=60, interval=30.0,
             noise=0.0, amb_noise=0.0, amb_step=None, t0=1_790_000_000.0, seed=1):
    r = random.Random(seed)
    readings = []
    t, T, stall_left = 0.0, start, stall_min * 60
    while t <= minutes * 60:
        amb = ambient + (amb_step[1] if amb_step and t >= amb_step[0] * 60 else 0.0)
        amb_r = round(amb + r.gauss(0, amb_noise), 1) if amb_noise else amb
        readings.append((t0 + t, round(T + (r.gauss(0, noise) if noise else 0.0), 1), amb_r))
        if stall_at is not None and T >= stall_at and stall_left > 0:
            T += 0.002 * interval / 60; stall_left -= interval
        else:
            T += k * (amb - T) * interval
        t += interval
    return readings

pipeline_cases = []
def add_pipeline(name, readings, target, cook, cuts):
    for n in cuts:
        rs = readings[:n]
        start = rs[0][1]
        feats = mp._build_features(rs, target, cook, start)
        row = [feats[kk] for kk in ORDER]
        pred = max(0.0, float(m.score(row)))
        pipeline_cases.append({"name": f"{name} n={n}", "target": target, "cook_name": cook,
                               "start_temp": start, "readings": [list(x) for x in rs],
                               "features": row, "minutes": pred})

add_pipeline("chicken breast oven 190C", simulate(190, 5, 74, 0.00012, minutes=70, noise=0.05, amb_noise=1.5, seed=2),
             74, "Poultry Chicken Breast Medium", [12, 30, 60, 90, 120])
add_pipeline("brisket smoker 115C with stall", simulate(115, 4, 96, 0.00012, stall_at=66, stall_min=120, minutes=600,
                                                        noise=0.05, amb_noise=2.0, seed=3),
             96, "Beef Brisket Fall Apart", [25, 200, 400, 520, 700, 1000, 1200])
add_pipeline("rib eye reverse sear 107C", simulate(107, 6, 54, 0.00010, minutes=90, noise=0.05, seed=4),
             54, "Beef Rib Eye Medium Rare", [20, 60, 100, 140])
add_pipeline("custom name, ambient step +40C at 30 min", simulate(120, 8, 70, 0.00008, minutes=120, amb_step=(30, 40), seed=5),
             70, "Custom", [40, 70, 100, 150, 200])
add_pipeline("category+cut with typed target (Lamb Leg, 60C)", simulate(160, 7, 60, 0.00009, minutes=90, seed=6),
             60.0, "Lamb Leg", [30, 90, 150])
add_pipeline("category only (Pork)", simulate(140, 5, 71, 0.00008, minutes=120, seed=7),
             71, "Pork", [50, 150])
add_pipeline("irregular gaps (sparse BLE)", [r for i, r in enumerate(simulate(150, 5, 63, 0.0001, minutes=80, seed=8)) if i % 7 not in (2, 3, 4)],
             63, "Pork Loin / Chop Medium", [15, 40, 70])

out = {
    "source": os.environ.get("PROBEABILITY_SOURCE", "https://github.com/snelstim/Probe-ability"),
    "feature_order": ORDER,
    "units": "degC, minutes, degC/min; readings are [unix_seconds, internal_C, ambient_C]",
    "note": "floats are Python repr (shortest round-trip); compare model_cases exactly or within 1e-9 (expm1)",
    "model_cases": model_cases,
    "pipeline_cases": pipeline_cases,
}
path = os.environ.get("PROBEABILITY_GOLDEN") or os.path.join(HERE, "probeability_golden.json")
with open(path, "w") as fh:
    json.dump(out, fh, indent=1)
print("wrote", path, len(model_cases), "model cases,", len(pipeline_cases), "pipeline cases")
for c in model_cases[::9]:
    print(f"  {c['name']:<45} {c['minutes']:.3f}")
for c in pipeline_cases:
    print(f"  {c['name']:<60} {c['minutes']}")
