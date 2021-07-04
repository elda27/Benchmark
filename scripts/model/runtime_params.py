from typing import Enum


class RuntimeParams(str, Enum):
    TrainingIndex = 'training_index'
    ValidationIndex = 'validation_index'
    TestIndex = 'test_index'
