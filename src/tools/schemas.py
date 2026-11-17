"""Argument schemas for the domain tools. Extra fields are rejected."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

OrderStatus = Literal[
    "searching_courier",
    "courier_assigned",
    "in_transit",
    "delivered",
    "cancelled_by_customer",
    "cancelled_no_courier",
]
VehicleType = Literal["motorcycle", "car", "van"]


class GetOrderStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(pattern=r"^A-[0-9]+$", max_length=32)


class ListMyOrdersInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: OrderStatus | None = None
    limit: int = Field(default=20, ge=1, le=20)


class SearchOrdersInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: str | None = Field(default=None, max_length=40)
    status: OrderStatus | None = None
    order_id: str | None = Field(default=None, pattern=r"^A-[0-9]+$", max_length=32)
    limit: int = Field(default=20, ge=1, le=20)

    @model_validator(mode="after")
    def at_least_one_filter(self) -> "SearchOrdersInput":
        if self.city is None and self.status is None and self.order_id is None:
            raise ValueError("provide city, status, or order_id")
        return self


class EstimateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distance_km: float = Field(gt=0, le=500)
    vehicle_type: VehicleType | None = None
    city: str | None = Field(default=None, max_length=40)
    hour_from: int | None = Field(default=None, ge=0, le=23)
    hour_to: int | None = Field(default=None, ge=0, le=23)

    @model_validator(mode="after")
    def hours_are_a_window(self) -> "EstimateInput":
        if (self.hour_from is None) != (self.hour_to is None):
            raise ValueError("hour_from and hour_to must be set together")
        if self.hour_from is not None and self.hour_from > self.hour_to:
            raise ValueError("hour_from must be less than or equal to hour_to")
        if self.city is not None:
            self.city = self.city.strip() or None
        return self


class SearchKnowledgeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=300)
    limit: int = Field(default=3, ge=1, le=3)


class CreateSupportTicketInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=500)
    order_id: str | None = Field(default=None, pattern=r"^A-[0-9]+$", max_length=32)
    confirmed: bool = False
    confirmation_id: str | None = Field(default=None, max_length=64)


class RequestCancellationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(pattern=r"^A-[0-9]+$", max_length=32)
    reason: str = Field(min_length=1, max_length=300)
    confirmed: bool = False
    confirmation_id: str | None = Field(default=None, max_length=64)
