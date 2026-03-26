import json
import urllib.request


class JSONRPC:
    def __init__(self):
        self.urlStr = ""

    def setURL(self, us: str):
        self.urlStr = us

    def request(self, req: dict) -> dict:
        req_bytes = json.dumps(req).encode("utf-8")
        request = urllib.request.Request(
            self.urlStr,
            data=req_bytes,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(req_bytes)),
                "Content-Language": "en-US",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as resp:
                body = resp.read().decode("utf-8")
            return json.loads(body)
        except Exception:
            return {}
