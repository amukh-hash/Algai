from fastapi import APIRouter, HTTPException
import pandas as pd
from app.models.data import DataRequest, DataResponse, PreprocessRequest
from app.data.synthetic import SyntheticDataHandler
from app.data.yfinance_loader import YFinanceHandler
from app.preprocessing.fractional import FractionalDifferencer

router = APIRouter()

@router.post("/ingest", response_model=DataResponse)
def ingest_data(request: DataRequest):
    """
    Ingest data from specified source.
    """
    if request.source == "synthetic":
        handler = SyntheticDataHandler(**request.params)
    elif request.source == "yfinance":
        handler = YFinanceHandler()
    else:
        raise HTTPException(status_code=400, detail="Unknown source")

    try:
        df = handler.fetch_data(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            timeframe=request.timeframe
        )

        # Convert index to column for JSON serialization
        df.reset_index(inplace=True)
        # Rename index column to lowercase 'date' if present
        df.rename(columns={'index': 'date', 'Date': 'date'}, inplace=True)

        data_records = df.to_dict(orient="records")
        return DataResponse(count=len(data_records), data=data_records)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/preprocess", response_model=DataResponse)
def preprocess_data(request: PreprocessRequest):
    """
    Apply preprocessing. Currently supports fractional differentiation.
    """
    try:
        df = pd.DataFrame(request.data)
        if 'date' in df.columns:
            df.set_index('date', inplace=True)

        if request.method == "fractional_diff":
            d = request.params.get("d", 0.4)
            thres = request.params.get("thres", 1e-4)
            diff = FractionalDifferencer(d=d, thres=thres)
            out_df = diff.transform(df)

            # Handle NaNs from diff
            out_df.dropna(inplace=True)

            out_df.reset_index(inplace=True)
            if 'index' in out_df.columns:
                out_df.rename(columns={'index': 'date'}, inplace=True)

            data_records = out_df.to_dict(orient="records")
            return DataResponse(count=len(data_records), data=data_records)
        else:
            raise HTTPException(status_code=400, detail="Unknown method")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
