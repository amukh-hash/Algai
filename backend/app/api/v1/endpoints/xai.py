from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Any, Dict
import torch
import numpy as np
from app.models.patchtst import HybridPatchTST
from app.xai.attribution import FeatureAttributor
from app.xai.visualization import extract_attention_weights

router = APIRouter()

class XAIRequest(BaseModel):
    input_data: List[List[float]] # (Lookback, Features)
    config: Dict[str, Any]

class XAIResponse(BaseModel):
    attributions: List[List[float]]
    attention: Dict[str, Any]

@router.post("/explain", response_model=XAIResponse)
def explain_prediction(request: XAIRequest):
    try:
        # Reconstruct model (In production, load from checkpoint)
        cfg = request.config
        model = HybridPatchTST(
            num_input_features=cfg.get('num_features', 5),
            lookback_window=cfg.get('lookback', 64),
            d_model=cfg.get('d_model', 64)
        )
        
        # Prepare input
        x_in = torch.tensor(request.input_data).unsqueeze(0).float() # (1, L, F)
        
        # Attribution
        attributor = FeatureAttributor(model)
        attrs = attributor.attribute(x_in)
        
        # Attention
        attn = extract_attention_weights(model, x_in)
        
        return XAIResponse(
            attributions=attrs.squeeze(0).tolist(),
            attention=attn
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
