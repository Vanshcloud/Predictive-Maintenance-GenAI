"""
src/api/routes/machines.py — Machine Metadata
==============================================

WHY: A dashboard needs to list what exists before it can show risk for it,
and that listing must not require scoring anything.
"""

from typing import List

from fastapi import APIRouter, HTTPException, status

from src.api.schemas import MachineInfo
from src.api.service import state

router = APIRouter(tags=["machines"])


def _require_store():
    if not state.store or not state.store.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dataset not loaded. Check /health.",
        )
    return state.store


@router.get("/machines", response_model=List[MachineInfo])
def list_machines() -> List[MachineInfo]:
    """Every machine this instance knows about, with data coverage."""
    store = _require_store()
    return [MachineInfo(**store.machine_info(m)) for m in store.machine_ids]


@router.get("/machines/{machine_id}", response_model=MachineInfo)
def get_machine(machine_id: int) -> MachineInfo:
    """Static facts about one machine."""
    store = _require_store()
    return MachineInfo(**store.machine_info(machine_id))
