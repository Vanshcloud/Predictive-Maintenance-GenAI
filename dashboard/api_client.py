"""
dashboard/api_client.py — HTTP Client for the Prediction API
=============================================================

WHY THIS FILE EXISTS:
    The dashboard is a **pure HTTP client**. It imports nothing from `src/` —
    no TensorFlow, no model, no pandas pipeline. That is deliberate: it is
    what lets the UI and the API be built, deployed, scaled, and restarted
    independently, and it is why this module re-declares its own small
    exception types instead of importing the project's hierarchy.

    If the dashboard imported `src.prediction`, containerising it would drag
    in TensorFlow and a 1.8 MB model file to render some charts.

HOW IT WORKS:
    One method per endpoint, each returning plain dicts. Every failure mode
    the UI must distinguish gets its own exception, because they need
    different words on screen:

      APIUnavailable  — nothing is listening. "Is the API running?"
      APIDegraded     — it answered, but cannot predict. "Model not loaded."
      APIError        — it answered with 4xx/5xx. Show the server's message.

    Collapsing these into one error would produce a dashboard that says
    "something went wrong" to a user who could otherwise be told exactly what
    to restart.
"""

from typing import Any, Dict, List, Optional

import requests

DEFAULT_BASE_URL = "http://localhost:8000"

# Predictions are ~140 ms; this is generous for those and for /fleet's cold
# path. Report generation gets its own, much longer, timeout.
DEFAULT_TIMEOUT = 30.0

# An LLM call takes ~21 s locally and the API caps itself at 120 s. The
# client must wait longer than the server's own ceiling, or it gives up on
# work that would have succeeded.
REPORT_TIMEOUT = 150.0


class APIUnavailable(Exception):
    """Nothing answered — wrong URL, or the API is not running."""


class APIDegraded(Exception):
    """The API is up but cannot serve predictions (model or data missing)."""


class APIError(Exception):
    """The API answered with an error status."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class APIClient:
    """Thin, synchronous client for the prediction API."""

    def __init__(
        self, base_url: str = DEFAULT_BASE_URL, timeout: float = DEFAULT_TIMEOUT
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------

    def _get(self, path: str, timeout: Optional[float] = None, **params) -> Any:
        return self._request("GET", path, timeout=timeout, params=params)

    def _post(self, path: str, payload: Dict, timeout: Optional[float] = None) -> Any:
        return self._request("POST", path, timeout=timeout, json=payload)

    def _request(self, method: str, path: str, timeout=None, **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(
                method, url, timeout=timeout or self.timeout, **kwargs
            )
        except requests.exceptions.ConnectionError as e:
            raise APIUnavailable(
                f"Could not reach the API at {self.base_url}. "
                f"Start it with `make run-api`. ({e.__class__.__name__})"
            ) from e
        except requests.exceptions.Timeout as e:
            raise APIUnavailable(
                f"The API at {self.base_url} did not respond within "
                f"{timeout or self.timeout:.0f}s."
            ) from e

        if response.status_code >= 400:
            raise APIError(self._describe(response), response.status_code)

        return response.json()

    @staticmethod
    def _describe(response) -> str:
        """
        Turn an error response into one sentence a human can act on.

        The API returns a structured body, but FastAPI's own validation errors
        use a different shape, so both are handled rather than assuming ours.
        """
        try:
            body = response.json()
        except ValueError:
            return f"HTTP {response.status_code}: {response.text[:200]}"

        detail = body.get("detail", body)
        if isinstance(detail, list):
            # Pydantic validation errors: a list of per-field problems.
            messages = [
                f"{'.'.join(str(p) for p in item.get('loc', [])[-2:])}: "
                f"{item.get('msg', '')}"
                for item in detail[:3]
            ]
            return "; ".join(messages)
        return str(detail)

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Liveness and readiness. Raises APIUnavailable if nothing answers."""
        return self._get("/health")

    def require_ready(self) -> Dict[str, Any]:
        """
        Fetch health and refuse to continue unless predictions can be served.

        Called before anything that needs a model, so the UI can say "the
        model is not loaded" rather than surfacing a 503 from three calls
        deeper.
        """
        health = self.health()
        if health.get("status") != "ok":
            raise APIDegraded(
                "The API is running but cannot serve predictions "
                f"(model_loaded={health.get('model_loaded')}, "
                f"dataset_loaded={health.get('dataset_loaded')}). "
                "Check the API logs."
            )
        return health

    def machines(self) -> List[Dict[str, Any]]:
        return self._get("/machines")

    def machine(self, machine_id: int) -> Dict[str, Any]:
        return self._get(f"/machines/{machine_id}")

    def predict(self, machine_id: int) -> Dict[str, Any]:
        return self._get(f"/machines/{machine_id}/predict")

    def explain(self, machine_id: int, history_hours: int = 24) -> Dict[str, Any]:
        return self._get(f"/machines/{machine_id}/explain", history_hours=history_hours)

    def history(self, machine_id: int, hours: int = 48) -> List[Dict[str, Any]]:
        return self._get(f"/machines/{machine_id}/history", hours=hours)

    def fleet(self, alerts_only: bool = False, refresh: bool = False) -> Dict[str, Any]:
        return self._get("/fleet", alerts_only=alerts_only, refresh=refresh)

    def report(
        self,
        machine_id: int,
        question: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a report or answer a question.

        Uses the long timeout: this is the one call that invokes a language
        model, and it takes tens of seconds.
        """
        payload: Dict[str, Any] = {"machine_id": machine_id}
        if question:
            payload["question"] = question
        if provider:
            payload["provider"] = provider
        if model:
            payload["model"] = model
        return self._post("/report", payload, timeout=REPORT_TIMEOUT)
