"""KOGS authorization formatting.

Credentials are caller-supplied; nothing in this module reads the environment
or stores secrets.
"""


def create_api_auth(api_key: str) -> str:
    """Return the canonical KOGS Authorization value for one API key.

    ``KOGS_API_KEY`` normally contains the raw 40-character key.  Accepting an
    already-prefixed value keeps older callers working, while validating here
    prevents both the previously observed missing-scheme request and accidental
    ``KSAT1-PLAIN KSAT1-PLAIN ...`` double-prefixing.
    """

    if not isinstance(api_key, str):
        raise TypeError("KOGS API key must be a string")
    value = api_key.strip()
    if value != api_key:
        raise ValueError("KOGS authorization value must not contain surrounding whitespace")
    prefix = "KSAT1-PLAIN "
    if value.startswith(prefix):
        token = value.removeprefix(prefix)
    elif any(character.isspace() for character in value):
        raise ValueError("unsupported KOGS authorization scheme")
    else:
        token = value
    if not token or any(character.isspace() for character in token):
        raise ValueError("KOGS API key must be one non-whitespace token")
    if len(token) != 40:
        raise ValueError("KOGS API key must contain exactly 40 characters")
    if not token.isascii() or not token.isprintable():
        raise ValueError("KOGS API key must contain printable ASCII characters")
    return f"{prefix}{token}"


def kogs_headers(api_key: str) -> dict[str, str]:
    """Return the request headers used by the read-only KOGS endpoints."""
    return {
        "Authorization": create_api_auth(api_key),
        "Accept": "application/json",
    }
