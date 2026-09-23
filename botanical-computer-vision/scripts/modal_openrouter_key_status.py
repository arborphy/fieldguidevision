"""Read non-secret OpenRouter key usage metadata from the Modal-held key."""

import modal


app = modal.App(
    "openrouter-key-status",
    image=modal.Image.debian_slim().pip_install("requests==2.32.5"),
)


@app.function(secrets=[modal.Secret.from_name("openrouter-bioimages-vlm")])
def key_status() -> dict:
    import os
    import requests

    response = requests.get(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json().get("data", {})
    return {
        key: data.get(key)
        for key in (
            "label",
            "limit",
            "limit_remaining",
            "usage",
            "usage_daily",
            "usage_weekly",
            "usage_monthly",
            "is_free_tier",
            "expires_at",
        )
    }


@app.local_entrypoint()
def main() -> None:
    print(key_status.remote())
