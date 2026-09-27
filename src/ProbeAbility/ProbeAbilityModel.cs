using System.Buffers.Binary;

namespace ProbeAbility;

/// <summary>
/// Probe-ability's cook-time model (github.com/snelstim/Probe-ability, MIT): two gradient-boosted tree
/// ensembles trained on ~217 real cooks, blended on degrees-to-go, answering "minutes until this probe
/// reaches its target".
///
/// <b>This is a port, not a reimplementation, and it is held to that.</b> The trees, tables and codes live
/// in the generated <c>ProbeAbilityModel.g.cs</c>; this file is their <c>score()</c>,
/// <c>_build_features()</c> and <c>resolve_meat()</c> line for line, and <c>ProbeAbilityTests</c> scores the
/// golden vectors their Python produced and fails on any difference. To take a new release, run
/// <c>tools/update.py</c> and then the tests - nothing here should need touching unless the
/// tests say their feature code changed.
///
/// <b>Feed it readings about 30 s apart</b>, as the integration does: "the last 3" and "the last 10"
/// ambient readings are counts, not windows, so a faster feed silently changes what they mean.
/// All temperatures are Celsius, rates degrees per minute, times seconds.
/// </summary>
public static partial class ProbeAbilityModel
{
    /// <summary>The integration's own debounce, which the count-based features assume.</summary>
    public static readonly TimeSpan ReadingInterval = TimeSpan.FromSeconds(30);

    /// <summary>Minutes remaining for one 17-feature row, never negative.</summary>
    public static double Predict(IReadOnlyList<Reading> readings, double target, string cookName, double start)
        => Math.Max(0.0, Score(Features(readings, target, cookName, start)));

    /// <summary><c>ml_model_code.score</c>: the log ensemble near target, the raw one far from it.</summary>
    public static double Score(ReadOnlySpan<double> x)
    {
        var near = Walk(Trees, Init, LearningRate, x);
        if (LogTarget) near = Math.Exp(near) - 1.0;
        if (!Blend) return near;

        var far = Walk(TreesFar, InitFar, LearningRateFar, x);
        var t = (RemainingFar - x[RemainingFeature]) / (RemainingFar - RemainingNear);
        t = t < 0.0 ? 0.0 : t > 1.0 ? 1.0 : t;
        var w = t * t * (3.0 - 2.0 * t);
        return (1.0 - w) * far + w * near;
    }

    /// <summary>Records are big-endian <c>&gt;hfhhf</c>: feature, threshold, left, right, value.</summary>
    private static double Walk((int Start, int Count)[] trees, double init, double rate, ReadOnlySpan<double> x)
    {
        var total = init;
        var nodes = Nodes.AsSpan();
        foreach (var (start, _) in trees)
        {
            var node = 0;
            while (true)
            {
                var record = nodes.Slice(start + node * NodeSize, NodeSize);
                var left = BinaryPrimitives.ReadInt16BigEndian(record[6..]);
                if (left == -1)
                {
                    total += rate * (double)BinaryPrimitives.ReadSingleBigEndian(record[10..]);
                    break;
                }

                var feature = BinaryPrimitives.ReadInt16BigEndian(record);
                var threshold = (double)BinaryPrimitives.ReadSingleBigEndian(record[2..]);
                node = x[feature] <= threshold ? left : BinaryPrimitives.ReadInt16BigEndian(record[8..]);
            }
        }

        return total;
    }

    /// <summary>One accepted reading: seconds (any epoch), probe and ambient Celsius.</summary>
    public readonly record struct Reading(double Seconds, double Internal, double Ambient);

    /// <summary><c>ml_predictor._build_features</c>, in <see cref="FeatureOrder"/>.</summary>
    public static double[] Features(IReadOnlyList<Reading> readings, double target, string cookName, double start)
    {
        var t0 = readings[0].Seconds;
        var times = readings.Select(r => r.Seconds - t0).ToArray();
        var internals = readings.Select(r => r.Internal).ToArray();
        var ambients = readings.Select(r => r.Ambient).ToArray();

        var rateInitial = Rate(times, internals, 600.0);
        var rateRecent = Rate(times, internals, 300.0);
        var deceleration = Math.Abs(rateInitial) > 0.01 ? rateRecent / rateInitial : 1.0;

        var current = internals[^1];
        var recent = ambients[^Math.Min(ambients.Length, 3)..];
        var ambientRecent = Sum(recent) / Math.Min(ambients.Length, 3);

        var inStall = Math.Abs(rateRecent) < 0.2 && current >= 40 && current <= 80 ? 1.0 : 0.0;

        var window = ambients.Length >= 10 ? ambients[^10..] : ambients;
        var mean = Sum(window) / window.Length;
        var std = window.Length > 1
            ? Math.Pow(Sum(window.Select(a => (a - mean) * (a - mean)).ToArray()) / window.Length, 0.5)
            : 0.0;

        var (category, animal, cutType, cut, preset) = ResolveMeat(cookName, target);

        return
        [
            current, ambientRecent, rateInitial, rateRecent, deceleration, start, target - current,
            times[^1] / 60.0, mean, std, inStall, ambientRecent - current,
            category, animal, cutType, cut, preset,
        ];
    }

    /// <summary><c>_compute_rate</c>: degrees per minute across the trailing window, or all readings.</summary>
    private static double Rate(double[] times, double[] temps, double windowSeconds)
    {
        if (times.Length < 2) return 0.0;
        var cutoff = times[^1] - windowSeconds;
        var first = Array.FindIndex(times, t => t >= cutoff);
        if (times.Length - first < 2) first = 0;
        var dt = times[^1] - times[first];
        return dt <= 0 ? 0.0 : (temps[^1] - temps[first]) / (dt / 60.0);
    }

    /// <summary>
    /// CPython 3.12+ <c>sum()</c> over floats is Neumaier-compensated, not a naive loop. The golden vectors
    /// were scored by it, and an ambient mean one ulp off can land a split the other way.
    /// </summary>
    private static double Sum(double[] values)
    {
        double sum = 0.0, compensation = 0.0;
        foreach (var value in values)
        {
            var next = sum + value;
            compensation += Math.Abs(sum) >= Math.Abs(value) ? (sum - next) + value : (value - next) + sum;
            sum = next;
        }

        return sum + compensation;
    }

    /// <summary>
    /// <c>resolve_meat</c>: a full preset name, else "Category Cut" with the doneness nearest the target,
    /// else a bare category, else the generic cook the model was trained to treat unknown names as.
    /// </summary>
    public static (int Category, int Animal, int CutType, int Cut, int Preset) ResolveMeat(string? cookName, double? target)
    {
        var name = (cookName ?? "").Trim();
        if (CookNames.TryGetValue(name, out var preset)) return preset;
        if (CutNames.TryGetValue(name, out var options))
        {
            // Python's min() keeps the first of equals, as MinBy does.
            return target is { } goal && options.Length > 0
                ? options.MinBy(o => Math.Abs(o.Celsius - goal)).Codes
                : CutNamesUntargeted[name];
        }

        return CategoryNames.TryGetValue(name, out var category) ? category : DefaultMeat;
    }
}
