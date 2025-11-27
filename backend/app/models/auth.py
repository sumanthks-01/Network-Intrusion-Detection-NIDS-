from pydantic import BaseModel

class TokenPayload(BaseModel):
    email: str | None = None
