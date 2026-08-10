"""Classification for unrecoverable OpenAI API failures."""


def is_fatal_api_error(exc):
    """Return whether retrying the API failure cannot recover the run."""
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code in {401, 403}:
        return True

    message = str(exc).lower()
    if any(keyword in message for keyword in (
        "余额",
        "insufficient balance",
        "balance is insufficient",
        "billing",
        "exceeded your current quota",
        "insufficient_quota",
        "quota_exceeded",
    )):
        return True
    if "error code: 401" in message or "error code: 403" in message:
        return True

    try:
        import openai

        return isinstance(
            exc,
            (openai.AuthenticationError, openai.PermissionDeniedError),
        )
    except Exception:
        return False
