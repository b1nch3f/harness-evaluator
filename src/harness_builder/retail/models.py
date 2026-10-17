"""Retail database models.

Field-for-field the same as tau2-bench's retail data model
(tau2/domains/retail/data_model.py at the pinned commit), so db.json loads
unchanged and tool results have the same JSON shape.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Variant(_Model):
    item_id: str
    options: Dict[str, str]
    available: bool
    price: float


class Product(_Model):
    name: str
    product_id: str
    variants: Dict[str, Variant]


class UserName(_Model):
    first_name: str
    last_name: str


class UserAddress(_Model):
    address1: str
    address2: str
    city: str
    country: str
    state: str
    zip: str


class CreditCard(_Model):
    source: Literal["credit_card"]
    id: str
    brand: str
    last_four: str


class Paypal(_Model):
    source: Literal["paypal"]
    id: str


class GiftCard(_Model):
    source: Literal["gift_card"]
    id: str
    balance: float


PaymentMethod = Union[CreditCard, GiftCard, Paypal]


class User(_Model):
    user_id: str
    name: UserName
    address: UserAddress
    email: str
    payment_methods: Dict[str, PaymentMethod]
    orders: List[str]


class OrderFullfilment(_Model):
    tracking_id: List[str]
    item_ids: List[str]


class OrderItem(_Model):
    name: str
    product_id: str
    item_id: str
    price: float
    options: Dict[str, str]


class OrderPayment(_Model):
    transaction_type: Literal["payment", "refund"]
    amount: float
    payment_method_id: str


OrderStatus = Literal[
    "processed",
    "pending",
    "pending (item modified)",
    "delivered",
    "cancelled",
    "exchange requested",
    "return requested",
]


class Order(_Model):
    order_id: str
    user_id: str
    address: UserAddress
    items: List[OrderItem]
    status: OrderStatus
    fulfillments: List[OrderFullfilment]
    payment_history: List[OrderPayment]
    cancel_reason: Optional[Literal["no longer needed", "ordered by mistake"]] = None
    exchange_items: Optional[List[str]] = None
    exchange_new_items: Optional[List[str]] = None
    exchange_payment_method_id: Optional[str] = None
    exchange_price_difference: Optional[float] = None
    return_items: Optional[List[str]] = None
    return_payment_method_id: Optional[str] = None


class RetailDB(_Model):
    products: Dict[str, Product]
    users: Dict[str, User]
    orders: Dict[str, Order]

    @classmethod
    def fresh(cls) -> "RetailDB":
        from harness_builder import data

        return cls.model_validate(data.db_json())

    def canonical(self) -> str:
        """Canonical JSON of the whole database, for comparisons."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True)

    def digest(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()
