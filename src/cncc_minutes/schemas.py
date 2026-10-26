from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Correction(StrictModel):
    id: str
    text: str
    uncertain: bool
    reason: str


class Corrections(StrictModel):
    segments: list[Correction]


class Evidence(StrictModel):
    text: str = Field(min_length=1, max_length=1800)
    sources: list[str] = Field(min_length=1, max_length=12)


class Action(StrictModel):
    task: str = Field(min_length=1, max_length=1000)
    owner: str
    deadline: str
    sources: list[str] = Field(min_length=1, max_length=12)


class Question(StrictModel):
    question: str
    answer: str
    sources: list[str] = Field(min_length=1, max_length=12)


class Term(StrictModel):
    name: str
    explanation: str
    sources: list[str] = Field(min_length=1, max_length=12)


class Summary(StrictModel):
    title: str = Field(min_length=1, max_length=160)
    overview: list[Evidence] = Field(max_length=8)
    key_points: list[Evidence] = Field(max_length=24)
    decisions: list[Evidence] = Field(max_length=12)
    actions: list[Action] = Field(max_length=12)
    questions: list[Question] = Field(max_length=16)
    terms: list[Term] = Field(max_length=16)
    uncertainties: list[Evidence] = Field(max_length=12)


def check_sources(summary: Summary, allowed: set[str]):
    for field in ("overview", "key_points", "decisions", "actions", "questions", "terms", "uncertainties"):
        for item in getattr(summary, field):
            if not set(item.sources) <= allowed:
                raise ValueError(f"{field} 引用了输入中不存在的片段 ID")
