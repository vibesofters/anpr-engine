"""Recognition V1 readiness status derived from repository data governance."""

from typing import Final, Literal

RecognitionV1Status = Literal[
    "READY_FOR_OWNER_TRAINING_AUTHORIZATION",
    "BLOCKED_ON_TRAINING_DATA",
]

RECOGNITION_V1_STATUS: Final[RecognitionV1Status] = "READY_FOR_OWNER_TRAINING_AUTHORIZATION"
TRAINING_AUTHORIZED_SAMPLE_COUNT: Final = 0
