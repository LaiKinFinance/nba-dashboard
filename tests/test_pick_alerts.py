"""Pick alerts: messages are encrypted so only the phone can read them, signed by the server, and sent once each:
a leg is done, the half-time check, and a close call in the 4th quarter."""
import json, struct
from contextlib import closing
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import app, live, push

def _phone():
    k = ec.generate_private_key(ec.SECP256R1()); auth = b"0123456789abcdef"
    pub = k.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    return k, push.b64u(pub), push.b64u(auth), pub, auth

def _decrypt(body, k, pub, auth):        # what the phone does (RFC 8291)
    salt, rs, idlen = body[:16], *struct.unpack(">IB", body[16:21]); as_pub = body[21:21 + idlen]; ct = body[21 + idlen:]
    shared = k.exchange(ec.ECDH(), ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), as_pub))
    ikm = push._hkdf_expand(push._hkdf_extract(auth, shared), b"WebPush: info\x00" + pub + as_pub, 32); prk = push._hkdf_extract(salt, ikm)
    pt = AESGCM(push._hkdf_expand(prk, b"Content-Encoding: aes128gcm\x00", 16)).decrypt(push._hkdf_expand(prk, b"Content-Encoding: nonce\x00", 12), ct, None)
    return pt[:-1]

def test_only_the_phone_can_read_the_message():
    k, p256, a, pub, auth = _phone()
    body = push.encrypt(b'{"title":"hi"}', p256, a)
    assert _decrypt(body, k, pub, auth) == b'{"title":"hi"}'

def test_messages_are_signed_by_the_server(tmp_path):
    key = push.load_keys(str(tmp_path / "k.pem")); assert push.load_keys(str(tmp_path / "k.pem")).private_numbers() == key.private_numbers()   # kept across restarts
    h = push.vapid_header(key, "https://fcm.googleapis.com/fcm/send/abc"); jwt = h.split("t=")[1].split(",")[0]
    head, body, sig = jwt.split("."); raw = push.unb64u(sig)
    key.public_key().verify(encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")), f"{head}.{body}".encode(), ec.ECDSA(hashes.SHA256()))
    assert json.loads(push.unb64u(body))["aud"] == "https://fcm.googleapis.com"

def test_alerts_done_half_and_close_each_once(monkeypatch):
    with closing(app.con()) as c:
        app._push_db(c); c.execute("DELETE FROM push_sent"); c.execute("DELETE FROM push_subs")
        c.execute("INSERT INTO push_subs VALUES('https://x/1','Mark','{}','{}','')")
        c.execute("DELETE FROM bets WHERE user='Mark'")
        legs = [dict(game_id="g1", player_id="emb", stat="reb", line=8.5, side="over", text="Joel Embiid"),
                dict(game_id="g1", player_id="jaq", stat="ast", line=4.5, side="over", text="Jaime Jaquez Jr."),
                dict(game_id="g1", stat="ml", side="home", line=0, text="76ers to win")]
        c.execute("INSERT INTO bets(user, created, legs, result) VALUES('Mark','2026-11-03', ?, 'open')", (json.dumps(legs),)); c.commit()
    sent = []
    monkeypatch.setattr(app, "_notify_user", lambda user, kind, title, body, tag: sent.append((kind, title, body)) or 1)
    monkeypatch.setattr(app, "projections_all", lambda: [])
    game = dict(game_id="g1", state="in", detail="Halftime", period=2, clock="0.0", home=dict(team="PHI", score=60), away=dict(team="CHI", score=52))
    box = {"teams": [{"team": "PHI", "players": [dict(player_id="emb", reb=9, min=18)]}, {"team": "MIA", "players": [dict(player_id="jaq", ast=1, min=16)]}]}
    monkeypatch.setattr(live, "scoreboard", lambda: [game]); monkeypatch.setattr(live, "boxscore", lambda gid: box)
    app.pick_alerts()
    kinds = [s[0] for s in sent]
    assert kinds.count("done") == 1 and "Embiid" in sent[0][2]                                   # 9 rebounds clears over 8.5
    half = next(s for s in sent if s[0] == "half"); assert half[1] == "Half-time: 2 of 3 on track" and "Jaquez" in half[2]
    app.pick_alerts(); assert len(sent) == 2                                                     # nothing is sent twice
    game.update(detail="5:30 - 4th", period=4, clock="5:30"); box["teams"][1]["players"][0].update(ast=3, min=30)
    app.pick_alerts()
    close = [s for s in sent if s[0] == "close"]; assert len(close) == 1 and "needs 2 more assists" in close[0][1] and "5:30 left in the 4th" in close[0][2]
