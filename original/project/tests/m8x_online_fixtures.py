"""CPU-only control-flow doubles copied from historical tests; never CUDA evidence."""
class FakeVector:
    def __init__(self, values): self.values = values
    def tolist(self): return self.values
class FakeDistribution:
    def __init__(self, values): self.probs = [FakeVector(values)]
class FakeChoice:
    def __init__(self, value): self.value = value
    def item(self): return self.value
class FakeTorch:
    float32 = "test-float32"
    def __init__(self): self.draws = []
    def tensor(self, values, **_): return FakeVector(values)
    def multinomial(self, probabilities, count, generator):
        self.draws.append((probabilities.tolist(), count, generator))
        generator.value += 1
        return FakeChoice(0)
class Stream:
    device = "cuda-test-double"
    def __init__(self, seed): self.value = 0; self.generator = self
    def snapshot(self): return self.value.to_bytes(8, "big")
    def restore(self, value): self.value = int.from_bytes(value, "big")
class FakePolicy:
    def verb_distribution(self, prepared): return FakeDistribution([1.0])
    def target_distribution(self, prepared, verbs): return FakeDistribution([1.0, 0.0])


