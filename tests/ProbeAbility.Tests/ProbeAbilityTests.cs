using System.Text.Json;
using ProbeAbility;

namespace ProbeAbility.Tests;

/// <summary>
/// The port of Probe-ability's model is held to its Python: every vector in <c>ProbeAbilityGolden.json</c>
/// was scored by their own code, and this suite fails on any difference. When
/// <c>tools/update.py</c> pulls a new release, a red test here means their feature code
/// changed and <c>ProbeAbilityModel.cs</c> must follow it; a green one means the new trees just work.
/// </summary>
public class ProbeAbilityTests
{
    private static readonly JsonDocument Golden = JsonDocument.Parse(
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "ProbeAbilityGolden.json")));

    public static TheoryData<int> ModelCases()
        => new(Enumerable.Range(0, Golden.RootElement.GetProperty("model_cases").GetArrayLength()));

    public static TheoryData<int> PipelineCases()
        => new(Enumerable.Range(0, Golden.RootElement.GetProperty("pipeline_cases").GetArrayLength()));

    private static double[] Doubles(JsonElement array) => array.EnumerateArray().Select(e => e.GetDouble()).ToArray();

    [Theory]
    [MemberData(nameof(ModelCases))]
    public void TheTreesScoreExactlyAsTheirPythonDoes(int index)
    {
        var @case = Golden.RootElement.GetProperty("model_cases")[index];
        var x = Doubles(@case.GetProperty("x"));
        var expected = @case.GetProperty("minutes").GetDouble();

        Assert.Equal(expected, ProbeAbilityModel.Score(x), 9);
    }

    [Theory]
    [MemberData(nameof(PipelineCases))]
    public void ReadingsBecomeTheSameFeaturesAndTheSamePrediction(int index)
    {
        var @case = Golden.RootElement.GetProperty("pipeline_cases")[index];
        var readings = @case.GetProperty("readings").EnumerateArray()
            .Select(r => new ProbeAbilityModel.Reading(r[0].GetDouble(), r[1].GetDouble(), r[2].GetDouble()))
            .ToList();
        var target = @case.GetProperty("target").GetDouble();
        var cook = @case.GetProperty("cook_name").GetString()!;
        var start = @case.GetProperty("start_temp").GetDouble();

        var features = ProbeAbilityModel.Features(readings, target, cook, start);
        var expected = Doubles(@case.GetProperty("features"));
        for (var i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - features[i]) <= 1e-9,
                $"{ProbeAbilityModel.FeatureOrder[i]}: expected {expected[i]:R}, got {features[i]:R}");

        if (@case.GetProperty("minutes").ValueKind == JsonValueKind.Number)
            Assert.Equal(@case.GetProperty("minutes").GetDouble(),
                ProbeAbilityModel.Predict(readings, target, cook, start), 9);
    }

    [Theory]
    [InlineData("Beef Brisket Fall Apart", 96.0)]
    [InlineData("Beef Rib Eye", 55.0)]
    [InlineData("Beef Rib Eye", null)]
    [InlineData("Pork", 70.0)]
    [InlineData("Custom", 70.0)]
    [InlineData("", null)]
    public void NamesResolveWithoutThrowing(string name, double? target)
        => Assert.InRange(ProbeAbilityModel.ResolveMeat(name, target).Cut, 0, 28);

    [Fact]
    public void TheGoldenVectorsCameFromTheModelThisBuildCarries()
        => Assert.Equal(ProbeAbilityModel.Source, Golden.RootElement.GetProperty("source").GetString());
}
