from fastapi import APIRouter, Header

router = APIRouter()

@router.get("/whoami")
async def whoami(authorization: str | None = Header(default=None)):
    return {"authorization": authorization}
