from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TopicNote(StrictModel):
    """Internal reading notes; these fields do not prescribe the delivered Markdown."""

    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=32000)
    segment_ids: list[str] = Field(min_length=1)


class ReadingNotes(StrictModel):
    topics: list[TopicNote]


class PlannedTopic(StrictModel):
    title: str = Field(min_length=1, max_length=200)
    focus: str = Field(min_length=1, max_length=1800)
    material_ids: list[str] = Field(min_length=1)


class TopicPlan(StrictModel):
    title: str = Field(min_length=1, max_length=200)
    introduction: str = Field(max_length=2000)
    topics: list[PlannedTopic] = Field(min_length=1)
