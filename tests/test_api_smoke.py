from fastapi.testclient import TestClient
from mvpfx.api import app

def test_endpoints_ok():
    c = TestClient(app)
    r1 = c.get("/signals"); assert r1.status_code == 200
    r2 = c.get("/explanations"); assert r2.status_code == 200
    r3 = c.post("/orders", json={"side":"long","qty":10000,"order_type":"MKT"})
    assert r3.status_code == 200
