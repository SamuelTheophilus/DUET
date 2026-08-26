from pydantic import BaseModel, Field


class SelectPromptStats(BaseModel):
    pool_size: int =  Field(default=0)
    skipped_easy: int = Field(default=0)
    skipped_hard: int = Field(default=0)
    selected: int = Field(default=0)
    considered: int = Field(default=0)
