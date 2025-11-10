from mvpfx.api import OrderRequest

def test_order_request_model():
    req = OrderRequest(side="long", qty=10000, order_type="MKT")
    assert req.side == "long" and req.qty == 10000
