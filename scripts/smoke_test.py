#!/usr/bin/env python3
"""Check API health and the Wine prediction response contract."""
import json
import math
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen

CLASSES = {"class_0", "class_1", "class_2"}


def main():
    url = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
    with urlopen(f"{url}/health", timeout=10) as response:
        if response.status != 200 or json.load(response) != {"status": "ok"}:
            raise ValueError("Unexpected health response")
    payload = {
        "alcohol": 13.2, "malic_acid": 1.8, "magnesium": 100.0,
        "flavanoids": 2.5, "color_intensity": 5.0, "proline": 900.0,
    }
    request = Request(f"{url}/predict", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        if response.status != 200:
            raise ValueError(f"Unexpected prediction status: {response.status}")
        result = json.load(response)
    if not isinstance(result, dict) or result.get("prediction") not in CLASSES:
        raise ValueError(f"Unknown predicted class: {result}")
    if "probabilities" in result:
        probabilities = result["probabilities"]
        if not isinstance(probabilities, dict) or set(probabilities) != CLASSES:
            raise ValueError("Expected a probability for each of the three classes")
        if not all(isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1
                   for p in probabilities.values()):
            raise ValueError("Probabilities must be finite values between zero and one")
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5):
            raise ValueError("Probabilities do not sum to one")
    print("API smoke test passed:", result)


if __name__ == "__main__":
    try:
        main()
    except (URLError, ValueError, TypeError, KeyError) as error:
        sys.exit(f"API smoke test failed: {error}")
