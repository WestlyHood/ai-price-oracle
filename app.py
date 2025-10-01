import os, time, json, numpy as np, asyncio, binascii
from fastapi import FastAPI
from eth_account import Account
from eth_account.messages import encode_defunct
from eth_utils import keccak
import httpx
from web3 import Web3
from collections import defaultdict, deque
from sklearn.linear_model import LinearRegression
from fastapi.middleware.cors import CORSMiddleware

# ----- CONFIG -----
ORACLE_PRIVKEY = os.getenv("ORACLE_PRIVKEY")
acct = Account.from_key(ORACLE_PRIVKEY)

# # RPC (Alchemy Sepolia in this case)
RPC_URL = os.getenv("RPC_URL")
w3 = Web3(Web3.HTTPProvider(RPC_URL))

CONTRACT_ADDRESS = Web3.to_checksum_address(
    os.getenv("CONTRACT_ADDRESS", "0xca042238b199cdaddd50824de2143b714324f01f")
)

# Load ABI from local file (save Remix ABI as AIPriceOracle.json)
with open("AIPriceOracle.json") as f:
    ABI = json.load(f)

contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI)

app = FastAPI()


# Enable CORS for frontend (Vercel + local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",              # local dev
        "https://oracle-dashboard-ochre.vercel.app/", 
        "*"                                   
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Interval for automation (seconds)
SUBMIT_INTERVAL = int(os.getenv("SUBMIT_INTERVAL", "30"))

# Cache for last results
latest_results = []

# Rolling history for predictions
price_history = defaultdict(lambda: deque(maxlen=20))


def to_e8(x): return int(round(x * 1e8))
def to_bp(x): return int(round(max(0, min(1, x)) * 10000))


# ----- FETCHERS -----
async def fetch_binance(symbol: str):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    async with httpx.AsyncClient() as c:
        r = await c.get(url)
        return float(r.json()["price"])


async def fetch_coinbase(product: str):
    url = f"https://api.exchange.coinbase.com/products/{product}/ticker"
    async with httpx.AsyncClient() as c:
        r = await c.get(url)
        return float(r.json()["price"])


async def fetch_coingecko(id_: str, vs: str):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={id_}&vs_currencies={vs}"
    async with httpx.AsyncClient() as c:
        r = await c.get(url)
        return float(r.json()[id_][vs])


# ----- PAIRS CONFIG -----
PAIRS = [
    {"base": "ETH", "quote": "USD", "binance": "ETHUSDT", "coinbase": "ETH-USD", "coingecko": ("ethereum", "usd")},
    {"base": "BTC", "quote": "USD", "binance": "BTCUSDT", "coinbase": "BTC-USD", "coingecko": ("bitcoin", "usd")},
    {"base": "ETH", "quote": "BNB", "binance": "ETHBNB", "coinbase": None, "coingecko": ("ethereum", "bnb")},
]


# ----- SAFE HEX HANDLER -----
def hex_to_bytes(sig: str) -> bytes:
    """Convert hex string (0x-prefixed) safely into bytes."""
    if sig.startswith("0x"):
        sig = sig[2:]
    try:
        return bytes.fromhex(sig)
    except binascii.Error:
        raise ValueError(f"Invalid signature format: {sig}")


# ----- AGGREGATOR -----
async def aggregate_price(pair):
    prices = []
    try: prices.append(await fetch_binance(pair["binance"]))
    except: pass

    if pair["coinbase"]:
        try: prices.append(await fetch_coinbase(pair["coinbase"]))
        except: pass

    try:
        id_, vs = pair["coingecko"]
        prices.append(await fetch_coingecko(id_, vs))
    except: pass

    if not prices:
        return None, None

    median = float(np.median(prices))
    std = float(np.std(prices)) if len(prices) > 1 else 0
    confidence = 1 - (std / (median + 1e-9))
    return median, confidence


# ----- PUSH TO CHAIN -----
# Track last gas bump globally
last_gas_bump = 0

def push_to_chain(base, quote, priceE8, confBP, ts, payload, sig):
    # Always get the latest *pending* nonce
    nonce = w3.eth.get_transaction_count(acct.address, "pending")

    # Use EIP-1559 style fees
    tx = contract.functions.submit(
        base, quote, priceE8, confBP, ts, payload, hex_to_bytes(sig)
    ).build_transaction({
        "from": acct.address,
        "nonce": nonce,
        "gas": 2_000_000,
        "maxFeePerGas": w3.to_wei("3", "gwei"),
        "maxPriorityFeePerGas": w3.to_wei("2", "gwei"),
    })

    signed_tx = w3.eth.account.sign_transaction(tx, ORACLE_PRIVKEY)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    return tx_hash.hex()

# ----- ENDPOINT -----
@app.get("/")
def home():
    return {"status": "ok", "message": "AI Oracle backend running", "endpoint": "/price"}


@app.get("/price")
async def get_price():
    """Return last cached results (auto-submitted)."""
    if not latest_results:
        return {"status": "warming_up", "message": "No submissions yet, please wait..."}
    return latest_results


# ----- AUTOMATION LOOP -----
async def auto_submit_loop():
    """Background loop that pushes all pairs every N seconds."""
    global latest_results
    while True:
        results = []
        try:
            ts = int(time.time())
            for pair in PAIRS:
                median, conf = await aggregate_price(pair)
                if median is None:
                    continue

                payload = f"AI_PRICE:{pair['base']}|{pair['quote']}|{to_e8(median)}|{to_bp(conf)}|{ts}"

                raw_hash = keccak(text=payload)
                msg = encode_defunct(primitive=raw_hash)
                signed = Account.sign_message(msg, ORACLE_PRIVKEY)
                sig = signed.signature.hex()
                if not sig.startswith("0x"):
                        sig = "0x" + sig


                # Push to chain
                tx_hash = push_to_chain(pair["base"], pair["quote"], to_e8(median), to_bp(conf), ts, payload, sig)

                # Save history for prediction
                key = f"{pair['base']}/{pair['quote']}"
                price_history[key].append((ts, median))

                # Prediction (5 min ahead)
                pred_price = "Not enough data yet"
                hist = price_history[key]
                if len(hist) >= 3:
                    times = np.array([h[0] for h in hist]).reshape(-1, 1)
                    prices = np.array([h[1] for h in hist])
                    model = LinearRegression().fit(times, prices)
                    future_time = ts + 300
                    pred_price = float(model.predict([[future_time]])[0])

                result = {
                    "base": pair["base"],
                    "quote": pair["quote"],
                    "priceE8": to_e8(median),
                    "confidenceBP": to_bp(conf),
                    "timestamp": ts,
                    "signer": acct.address,
                    "payload": payload,
                    "signature": sig,
                    "txHash": tx_hash,
                    "prediction5m": pred_price,
                }
                results.append(result)

                print(f"[AutoSubmit] {pair['base']}/{pair['quote']}={median:.2f} "
                      f"Conf={conf:.4f} tx={tx_hash} pred5m={pred_price}")

            latest_results = results
        except Exception as e:
            print(f"[AutoSubmit] Error: {e}")

        await asyncio.sleep(SUBMIT_INTERVAL)


@app.on_event("startup")
async def startup_event():
    """Kick off background auto-submit loop when FastAPI starts."""
    asyncio.create_task(auto_submit_loop())
