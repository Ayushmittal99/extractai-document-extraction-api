"""All Pydantic models: API request/response + the extraction schemas Gemini must follow.

Convention: missing scalar -> null (None); missing list -> [].
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema, field_validator

# Money is parsed as Decimal (no float rounding errors) but described to Gemini as a plain JSON number.
Money = Annotated[Decimal, WithJsonSchema({"type": "number"})]


class DocumentType(str, Enum):
    resume = "resume"
    invoice = "invoice"


# ---------------------------------------------------------------- resume
class Education(BaseModel):
    degree: str | None = None
    field_of_study: str | None = None
    institution: str | None = None
    year: str | None = Field(None, description="Graduation year or period exactly as written, e.g. '2024'")


class WorkExperience(BaseModel):
    job_title: str | None = None
    company: str | None = None
    start_date: str | None = Field(None, description="As written in the text, e.g. 'Jan 2022'")
    end_date: str | None = Field(None, description="As written, e.g. 'Mar 2024' or 'Present'")
    duration_months: int | None = Field(None, ge=0, description="Only if stated or computable from explicit dates")
    description: str | None = None


class Project(BaseModel):
    name: str | None = None
    description: str | None = None
    technologies: list[str] = Field(default_factory=list)


class ResumeData(BaseModel):
    model_config = ConfigDict(extra="ignore")  # unexpected model fields are dropped (and reported as warnings)

    full_name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    skills: list[str] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    job_titles: list[str] = Field(default_factory=list)
    previous_companies: list[str] = Field(default_factory=list)
    experience_months: int | None = Field(None, ge=0, description="Total experience; null unless reliably supported")
    certifications: list[str] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)

    @field_validator("skills", "job_titles", "previous_companies", "certifications", mode="before")
    @classmethod
    def _none_to_empty_list(cls, v):
        return [] if v is None else v  # models sometimes send null for an empty list


# ---------------------------------------------------------------- invoice
class LineItem(BaseModel):
    description: str | None = None
    quantity: Money | None = None
    unit_price: Money | None = None
    amount: Money | None = None


class InvoiceData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    invoice_number: str | None = None
    invoice_date: date | None = Field(None, description="ISO format YYYY-MM-DD")
    due_date: date | None = Field(None, description="ISO format YYYY-MM-DD")
    vendor_name: str | None = None
    customer_name: str | None = None
    currency: str | None = Field(None, description="ISO code (e.g. USD, INR) or symbol exactly as written; null if absent")
    subtotal: Money | None = None
    tax_amount: Money | None = None
    total_amount: Money | None = None
    line_items: list[LineItem] = Field(default_factory=list)

    @field_validator("line_items", mode="before")
    @classmethod
    def _none_to_empty_list(cls, v):
        return [] if v is None else v


# ---------------------------------------------------------------- API
class ExtractionRequest(BaseModel):
    document_type: DocumentType
    text: str = Field(..., description="Unstructured resume or invoice text")

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty or whitespace-only")
        return v


class _BaseResponse(BaseModel):
    status: Literal["success", "success_with_warnings"]
    warnings: list[str] = Field(default_factory=list)


class ResumeResponse(_BaseResponse):
    document_type: Literal["resume"] = "resume"
    data: ResumeData


class InvoiceResponse(_BaseResponse):
    document_type: Literal["invoice"] = "invoice"
    data: InvoiceData


ExtractionResponse = Annotated[Union[ResumeResponse, InvoiceResponse], Field(discriminator="document_type")]


class ErrorResponse(BaseModel):
    error: str  # machine-readable code, e.g. "rate_limited"
    message: str  # safe, human-readable
