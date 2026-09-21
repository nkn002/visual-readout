from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException, Response
from sqlmodel import select

from ..database import SessionDep
from ..dependencies import SampleDep
from ..models import Annotation, Sample
from ..schemas import AnnotationCreate, AnnotationRead, AnnotationUpdate, ImportResult

router = APIRouter(prefix="/api", tags=["annotations"])


@router.get("/annotations", response_model=list[AnnotationRead])
def export_annotations(session: SessionDep) -> list[Annotation]:
    return list(session.exec(select(Annotation).order_by(Annotation.created_at)).all())


@router.get("/samples/{sample_id}/annotations", response_model=list[AnnotationRead])
def list_annotations(sample: SampleDep, session: SessionDep) -> list[Annotation]:
    stmt = select(Annotation).where(Annotation.sample_id == sample.id).order_by(Annotation.created_at)
    return list(session.exec(stmt).all())


@router.post("/samples/{sample_id}/annotations", response_model=AnnotationRead, status_code=201)
def create_annotation(sample: SampleDep, body: AnnotationCreate, session: SessionDep) -> Annotation:
    if body.sample_id != sample.id:
        raise HTTPException(422, "sample_id in body does not match the URL")
    ann_id = body.id or str(uuid4())
    if session.get(Annotation, ann_id) is not None:
        raise HTTPException(409, f"annotation {ann_id!r} already exists")
    row = Annotation(id=ann_id, sample_id=sample.id, label=body.label, note=body.note, geometry=body.geometry, source=body.source,
                     **({"created_at": body.created_at} if body.created_at else {}))
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


@router.patch("/annotations/{annotation_id}", response_model=AnnotationRead)
def update_annotation(annotation_id: str, body: AnnotationUpdate, session: SessionDep) -> Annotation:
    """Change the label or note of an existing annotation; the box itself stays as drawn."""
    row = session.get(Annotation, annotation_id)
    if row is None:
        raise HTTPException(404, f"annotation {annotation_id!r} not found")
    if body.label is not None:
        row.label = body.label
    if body.note is not None:
        row.note = body.note
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


@router.delete("/annotations/{annotation_id}", status_code=204)
def delete_annotation(annotation_id: str, session: SessionDep) -> Response:
    row = session.get(Annotation, annotation_id)
    if row is None:
        raise HTTPException(404, f"annotation {annotation_id!r} not found")
    session.delete(row)
    session.commit()
    return Response(status_code=204)


@router.post("/annotations/import", response_model=ImportResult)
def import_annotations(body: list[AnnotationCreate], session: SessionDep) -> ImportResult:
    """Bulk insert, for annotations exported from static mode. Existing ids and unknown samples are skipped."""
    imported = skipped = 0
    for a in body:
        if not a.id or session.get(Annotation, a.id) is not None or session.get(Sample, a.sample_id) is None:
            skipped += 1
            continue
        session.add(Annotation(id=a.id, sample_id=a.sample_id, label=a.label, note=a.note, geometry=a.geometry, source=a.source,
                               **({"created_at": a.created_at} if a.created_at else {})))
        imported += 1
    session.commit()
    return ImportResult(imported=imported, skipped=skipped)
