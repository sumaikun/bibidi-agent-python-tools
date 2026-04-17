"""
Routes: /detect, /see, /embed, /find
"""

from fastapi import APIRouter, HTTPException

from vision_api.schemas import DetectRequest, SeeRequest, EmbedRequest, FindRequest, DeepSearchRequest, CompareRequest
from vision_api.services import detect as detect_svc
from vision_api.services import see as see_svc
from vision_api.services import embed as embed_svc
from vision_api.services import deep_search as deep_search_svc
from vision_api.services import compare as compare_svc

router = APIRouter()


@router.post("/detect")
async def detect(req: DetectRequest):
    return await detect_svc.detect(req.screenshot, req.url)
 
 
@router.post("/see")
async def see(req: SeeRequest):
    return  see_svc.see(req.screenshot, req.question, req.url)
 
 
@router.post("/embed")
def embed(req: EmbedRequest):
    return embed_svc.embed(req.url, req.elements)
 
 
@router.post("/find")
def find(req: FindRequest):
    result = embed_svc.find(req.url, req.query, req.limit)
    if result is None:
        raise HTTPException(404, f"No elements cached for URL: {req.url} — call /embed first")
    return result

@router.post("/deep_search")
async def deep_search(req: DeepSearchRequest):
    return await deep_search_svc.deep_search(req.screenshot, req.question, req.url)


@router.post("/compare")
def compare(req: CompareRequest):
    return compare_svc.compare(req.before, req.after, req.expectation)