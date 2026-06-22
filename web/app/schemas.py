"""Pydantic request/response schemas for the JSON API surface."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class SignupIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=200)


class UserOut(BaseModel):
    id: uuid.UUID
    email: EmailStr
    name: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ProgressIn(BaseModel):
    lesson_id: str = Field(min_length=1, max_length=64)
    status: str = Field(default="completed", max_length=20)


class ProgressOut(BaseModel):
    lesson_id: str
    status: str
    completed_at: datetime

    model_config = {"from_attributes": True}
