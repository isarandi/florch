class NonfiniteDuringTrainingError(Exception):
    def __init__(self, name, tensor):
        super().__init__()
        self.name = name
        self.tensor = tensor

    def __str__(self):
        return f"Non-finite values in {self.name}:\n{self.tensor}"
