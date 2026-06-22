"""Lesson progress: list user's completed lessons, mark a lesson complete."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import require_user
from ..models import Progress, User
from ..schemas import ProgressIn, ProgressOut

router = APIRouter(prefix="/api/progress", tags=["progress"])


@router.get("", response_model=list[ProgressOut])
def list_progress(user: User = Depends(require_user), db: Session = Depends(get_db)):
    rows = db.query(Progress).filter(Progress.user_id == user.id).order_by(Progress.completed_at.desc()).all()
    return [ProgressOut.model_validate(r) for r in rows]


@router.post("", response_model=ProgressOut, status_code=status.HTTP_201_CREATED)
def mark_complete(payload: ProgressIn, user: User = Depends(require_user), db: Session = Depends(get_db)):
    existing = db.query(Progress).filter(Progress.user_id == user.id, Progress.lesson_id == payload.lesson_id).first()
    if existing:
        return ProgressOut.model_validate(existing)
    row = Progress(user_id=user.id, lesson_id=payload.lesson_id, status=payload.status)
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="already recorded")
    db.refresh(row)
    return ProgressOut.model_validate(row)
