"""
REST API Package
================
FastAPI application exposing predictions and LLM-generated reports.

Prediction endpoints never call an LLM — see `routes/reports.py` for why the
slow path is isolated.
"""
