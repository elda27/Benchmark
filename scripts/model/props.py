from typing import Generic, List, Optional, TypeVar
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel

T = TypeVar('T')


class TrainingProps(BaseModel):
    index_file: Path
    target_columns: List[str]
    drop_columns: List[str]

    dataset_query: Optional[str] = Field(
        None,
        help="filtering dataset by using DataFrame.query before processing."
    )

    preprocess: Optional[str] = None


class Props(GenericModel, Generic[T]):
    training: TrainingProps
    model: T
