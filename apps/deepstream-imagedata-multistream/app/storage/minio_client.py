from __future__ import annotations

import datetime as dt
import hashlib
import hmac
from dataclasses import dataclass
from urllib.parse import quote, urlsplit

import requests


def _sign(key: bytes, value: str) -> bytes:
    return hmac.new(key, value.encode(), hashlib.sha256).digest()


@dataclass
class MinioClient:
    endpoint: str
    access_key: str
    secret_key: str
    region: str = "us-east-1"

    def put(
        self, bucket: str, object_key: str, payload: bytes, content_type: str
    ) -> str:
        parsed = urlsplit(self.endpoint)
        now = dt.datetime.now(dt.timezone.utc)
        stamp, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
        path = f"/{quote(bucket)}/{quote(object_key, safe='/')}"
        digest = hashlib.sha256(payload).hexdigest()
        headers = {
            "host": parsed.netloc,
            "x-amz-content-sha256": digest,
            "x-amz-date": stamp,
            "content-type": content_type,
        }
        signed_headers = ";".join(sorted(headers))
        canonical_headers = "".join(
            f"{key}:{headers[key]}\n" for key in sorted(headers)
        )
        canonical = (
            f"PUT\n{path}\n\n{canonical_headers}\n{signed_headers}\n{digest}"
        )
        scope = f"{day}/{self.region}/s3/aws4_request"
        to_sign = (
            "AWS4-HMAC-SHA256\n"
            f"{stamp}\n{scope}\n"
            f"{hashlib.sha256(canonical.encode()).hexdigest()}"
        )
        key = _sign(
            _sign(
                _sign(_sign(("AWS4" + self.secret_key).encode(), day),
                      self.region),
                "s3",
            ),
            "aws4_request",
        )
        signature = hmac.new(
            key, to_sign.encode(), hashlib.sha256
        ).hexdigest()
        headers["authorization"] = (
            "AWS4-HMAC-SHA256 "
            f"Credential={self.access_key}/{scope},"
            f"SignedHeaders={signed_headers},Signature={signature}"
        )
        response = requests.put(
            f"{parsed.scheme}://{parsed.netloc}{path}",
            data=payload,
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        return response.headers.get("ETag", "").strip('"')
