"""Phone notifications (Web Push) without any outside service: the server signs each message with its own key (VAPID) and encrypts it for
the phone (RFC 8291, aes128gcm). The key pair is made once and kept on the data volume, so phones stay subscribed across restarts."""
import base64, json, os, struct, time
from urllib.parse import urlparse
import requests
from cryptography.hazmat.primitives import hashes, hmac, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

SUBJECT = "https://nba-dashboard.fly.dev"


def b64u(b): return base64.urlsafe_b64encode(b).rstrip(b"=").decode()
def unb64u(s): return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _raw_public(key): return key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)


def load_keys(path):
    """The server's signing key, made on first use and kept at `path` (PEM)."""
    if os.path.exists(path):
        with open(path, "rb") as f: return serialization.load_pem_private_key(f.read(), None)
    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f: f.write(pem)
    os.replace(tmp, path)
    return key


def public_key_b64(key): return b64u(_raw_public(key))


def vapid_header(key, endpoint):
    u = urlparse(endpoint)
    head = b64u(json.dumps({"typ": "JWT", "alg": "ES256"}, separators=(",", ":")).encode())
    body = b64u(json.dumps({"aud": f"{u.scheme}://{u.netloc}", "exp": int(time.time()) + 12 * 3600, "sub": SUBJECT}, separators=(",", ":")).encode())
    r, s = decode_dss_signature(key.sign(f"{head}.{body}".encode(), ec.ECDSA(hashes.SHA256())))
    jwt = f"{head}.{body}.{b64u(r.to_bytes(32, 'big') + s.to_bytes(32, 'big'))}"
    return f"vapid t={jwt}, k={public_key_b64(key)}"


def _hkdf_extract(salt, ikm):
    h = hmac.HMAC(salt, hashes.SHA256()); h.update(ikm); return h.finalize()


def _hkdf_expand(prk, info, n):
    h = hmac.HMAC(prk, hashes.SHA256()); h.update(info + b"\x01"); return h.finalize()[:n]


def encrypt(payload: bytes, p256dh: str, auth: str, salt=None, server_key=None):
    """aes128gcm body for one phone (RFC 8291 / RFC 8188, a single record)."""
    ua_pub = unb64u(p256dh); secret = unb64u(auth)
    server_key = server_key or ec.generate_private_key(ec.SECP256R1()); as_pub = _raw_public(server_key)
    shared = server_key.exchange(ec.ECDH(), ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), ua_pub))
    ikm = _hkdf_expand(_hkdf_extract(secret, shared), b"WebPush: info\x00" + ua_pub + as_pub, 32)
    salt = salt or os.urandom(16); prk = _hkdf_extract(salt, ikm)
    cek = _hkdf_expand(prk, b"Content-Encoding: aes128gcm\x00", 16); nonce = _hkdf_expand(prk, b"Content-Encoding: nonce\x00", 12)
    return salt + struct.pack(">IB", 4096, len(as_pub)) + as_pub + AESGCM(cek).encrypt(nonce, payload + b"\x02", None)


def send(key, sub, message: dict, ttl=3600):
    """Send one notification. Returns the HTTP status; 404/410 mean the phone has unsubscribed and the subscription should be dropped."""
    body = encrypt(json.dumps(message).encode(), sub["keys"]["p256dh"], sub["keys"]["auth"])
    r = requests.post(sub["endpoint"], data=body, timeout=15, headers={
        "TTL": str(ttl), "Content-Encoding": "aes128gcm", "Content-Type": "application/octet-stream", "Urgency": "high",
        "Authorization": vapid_header(key, sub["endpoint"])})
    return r.status_code
