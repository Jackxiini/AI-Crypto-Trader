import os
from dotenv import load_dotenv
load_dotenv()
OKX_API_KEY=os.getenv("OKX_API_KEY","")
OKX_SECRET=os.getenv("OKX_SECRET","")
OKX_PASSWORD=os.getenv("OKX_PASSWORD","")
OKX_TESTNET=os.getenv("OKX_TESTNET","true").lower()=="true"
PORT=int(os.getenv("PORT","8000"))
