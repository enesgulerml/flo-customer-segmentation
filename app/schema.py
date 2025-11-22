from pydantic import BaseModel, Field


# Data from the user
class CustomerInput(BaseModel):
    # Recency
    recency_days: int = Field(..., gt=0, description="Number of days since the last purchase")

    # Frequency
    total_orders: int = Field(..., gt=0, description="Total number of orders")

    # Monetary
    total_price: float = Field(..., gt=0, description="Total spending amount")

    # Tenure
    tenure_days: int = Field(..., gt=0, description="Number of days since first purchase")

    class Config:
        json_schema_extra = {
            "example": {
                "recency_days": 30,
                "total_orders": 5,
                "total_price": 2500.50,
                "tenure_days": 500
            }
        }


# The result will exist in API
class PredictionResponse(BaseModel):
    cluster_id: int
    cluster_name: str
    model_version: str