# probeability-dotnet

A .NET port of the cook-time model from [Probe-ability](https://github.com/snelstim/Probe-ability) by
Tim Snels: two gradient-boosted tree ensembles, trained on a few hundred real cooks (MEATER exports and
community-shared ones), that answer "how many minutes until this probe reaches its target?"

All the credit for the model goes to Probe-ability. This repo only makes it callable from C#, and
**proves the port is exact**: the test suite scores more than a hundred golden vectors that Probe-ability's
own Python produced and fails on any difference.

```csharp
using ProbeAbility;

// One reading per ~30 s, as the model was trained: seconds, probe °C, ambient (pit/oven) °C.
var readings = new List<ProbeAbilityModel.Reading> { new(t0, 21.0, 175.0), /* ... */ };
double minutes = ProbeAbilityModel.Predict(readings, target: 74.0, cookName: "Poultry Chicken Breast Medium", start: 21.0);
```

`cookName` uses Probe-ability's preset names ("Beef Brisket Fall Apart", "Pork Shoulder Pulled", ...);
"Category Cut" with a target, or a bare category, also resolve, and anything else is treated as a generic
cook exactly as the model was trained to.

**Feed it readings about 30 s apart.** Some features count readings ("the last 3", "the last 10") rather
than measuring time, so a faster feed quietly changes what they mean. Probe-ability's own integration
also only trusts the model after 10 readings spanning 10 minutes.

## How it stays current

- `tools/update.py [--ref TAG]` clones a Probe-ability release, regenerates
  `src/ProbeAbility/ProbeAbilityModel.g.cs` (the model blob, tables and encodings, carrying the upstream
  licence) and `tests/ProbeAbility.Tests/ProbeAbilityGolden.json` (vectors scored by upstream's Python).
- `.github/workflows/update.yml` runs that weekly and opens a pull request when a new tag appears. CI
  runs the golden tests on it. **Green** means the new trees just work. **Red** means upstream changed
  its feature code, and `ProbeAbilityModel.cs` has to follow it by hand.

## What is ported and what is not

Ported line for line: `score()`, `_build_features()`, `_compute_rate()` and `resolve_meat()`, including
CPython's compensated `sum()`. An ambient mean one ulp off can land a tree split the other way.

Not ported: the integration around the model (debounce, stall and rest phases, smoothing, the Newton
fallback, confidence). Those are behaviour choices for whoever calls it.

## Licence

MIT. See [LICENSE](LICENSE); the generated file also carries Probe-ability's original notice.
