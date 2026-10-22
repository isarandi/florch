class UnexpectedValueDuringTrainingError(Exception):
    def __init__(self, name, tensor):
        super().__init__()
        self.name = name
        self.tensor = tensor

    def __str__(self):
        return f"Unexpected value in {self.name}:\n{self.tensor}"


# Backward compatibility alias
NonfiniteDuringTrainingError = UnexpectedValueDuringTrainingError
