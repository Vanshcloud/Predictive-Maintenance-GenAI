"""
src/utils/exceptions.py — Custom Exception Hierarchy
============================================================

WHY THIS FILE EXISTS:
    Using generic exceptions (ValueError, RuntimeError) everywhere
    makes error handling imprecise. You end up catching errors you
    didn't intend to catch, or you can't distinguish between
    "invalid sensor data" and "model file not found."

    Custom exceptions let you:
    1. Catch SPECIFIC errors in SPECIFIC places
    2. Add context (which equipment? which sensor?)
    3. Return meaningful error messages to API clients
    4. Log different error types differently

DESIGN PATTERN:
    Exception hierarchy — all custom exceptions inherit from a
    single base class (PredMaintenanceError). This lets you:
    - Catch ALL project errors: except PredMaintenanceError
    - Catch SPECIFIC errors: except ModelNotFoundError
    - Add new exceptions without breaking existing handlers

USAGE:
    from src.utils.exceptions import DataValidationError

    if df.empty:
        raise DataValidationError(
            message="Sensor data is empty",
            details={"file": "sensors_2024.csv", "expected_rows": 1000}
        )
"""

from typing import Any, Dict, Optional


class PredMaintenanceError(Exception):
    """
    Base exception for the Predictive Maintenance application.

    All custom exceptions inherit from this class.
    This allows catching ALL project-specific errors with a single
    except clause when needed.
    """

    def __init__(
        self,
        message: str = "An error occurred in the Predictive Maintenance system",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# ---- Data Layer Exceptions ----


class DataIngestionError(PredMaintenanceError):
    """Raised when data cannot be loaded from the source."""

    def __init__(
        self,
        message: str = "Failed to ingest data",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, details=details)


class DataValidationError(PredMaintenanceError):
    """Raised when data fails quality or schema validation checks."""

    def __init__(
        self,
        message: str = "Data validation failed",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, details=details)


class DataPreprocessingError(PredMaintenanceError):
    """Raised when data preprocessing or feature engineering fails."""

    def __init__(
        self,
        message: str = "Data preprocessing failed",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, details=details)


# ---- Model Layer Exceptions ----


class ModelNotFoundError(PredMaintenanceError):
    """Raised when a saved model file cannot be located."""

    def __init__(
        self,
        message: str = "Model artifact not found",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, details=details)


class ModelTrainingError(PredMaintenanceError):
    """Raised when model training encounters an unrecoverable error."""

    def __init__(
        self,
        message: str = "Model training failed",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, details=details)


class PredictionError(PredMaintenanceError):
    """Raised when the prediction pipeline fails."""

    def __init__(
        self,
        message: str = "Prediction failed",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, details=details)


# ---- GenAI Layer Exceptions ----


class LLMConnectionError(PredMaintenanceError):
    """Raised when the LLM provider cannot be reached."""

    def __init__(
        self,
        message: str = "Failed to connect to LLM provider",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, details=details)


class ReportGenerationError(PredMaintenanceError):
    """Raised when GenAI report generation fails."""

    def __init__(
        self,
        message: str = "Report generation failed",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, details=details)


# ---- API Layer Exceptions ----


class APIError(PredMaintenanceError):
    """Base exception for API-layer errors."""

    def __init__(
        self,
        message: str = "API error occurred",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.status_code = status_code
        super().__init__(message=message, details=details)


class ResourceNotFoundError(APIError):
    """Raised when a requested resource does not exist (HTTP 404)."""

    def __init__(
        self,
        message: str = "Resource not found",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, status_code=404, details=details)
