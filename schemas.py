"""
Pydantic request/response schemas for all endpoints.
"""

from pydantic import BaseModel


# --- Vision ---

class DetectRequest(BaseModel):
    screenshot: str
    url: str

class SeeRequest(BaseModel):
    screenshot: str
    question: str
    url: str | None = None  # if provided, runs detect first and sends annotated image to VLM

class EmbedRequest(BaseModel):
    url: str
    elements: list

class FindRequest(BaseModel):
    url: str
    query: str
    limit: int = 5
    
class DeepSearchRequest(BaseModel):
    screenshot: str
    question: str
    url: str


class CompareRequest(BaseModel):
    before: str              
    after: str               
    expectation: str

# --- SAM3 Segment ---

class SegmentRequest(BaseModel):
    screenshot: str
    prompts: list[str]
    save_annotated: bool = True
    grid: dict | None = None


# --- CAPTCHA ---

class DetectGridRequest(BaseModel):
    screenshot: str

class SolveRequest(BaseModel):
    screenshot: str
    object_hint: str = ""
    rows: int = 0
    cols: int = 0
    min_overlap: int = 20