from __future__ import annotations
import os
import urllib.parse
from ....registry import Command

_TWELVEDATA_ENDPOINT = "https://api.twelvedata.com/time_series"

def _build_query(symbol: str, interval: str, outputsize: int, apikey: str | None):
    q = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": str(outputsize),
        "format": "JSON",
    }
    if apikey:  # only attach if present
        q["apikey"] = apikey
    return q

def _encode_url(base: str, params: dict) -> str:
    return f"{base}?{urllib.parse.urlencode(params)}"

class Request(Command):
    name = "request"
    aliases = ["req"]
    mode = "chart"
    help = "Show the TwelveData request for the current ticker/interval/outputsize. Usage: request [--show-key]"

    def run(self, args, state) -> None:
        if state.mode != "chart":
            print("Error: set mode to 'chart' first (use: mode chart | mc).")
            return
        if not state.ticker:
            print("Error: ticker not set. Use: t SYMBOL or tSYMBOL[:interval[:outputsize]]")
            return

        show_key = any(a in ("--show-key", "--showkey", "-k") for a in args)
        apikey_env = os.getenv("TWELVE_DATA_API_KEY")

        apikey_used = apikey_env if show_key else ("<REDACTED>" if apikey_env else None)
        q = _build_query(state.ticker, state.interval, state.outputsize, apikey_used)
        url = _encode_url(_TWELVEDATA_ENDPOINT, q)

        # Print a friendly summary
        print("TwelveData time_series request")
        print("-------------------------------")
        print(f"symbol     : {state.ticker}")
        print(f"interval   : {state.interval}")
        print(f"outputsize : {state.outputsize}")
        print(f"api key    : {'present' if apikey_env else 'MISSING'}{' (shown)' if show_key and apikey_env else ''}")
        print()
        print("GET URL:")
        print(url)
        print()
        # curl with safe quoting; omit apikey entirely if not present
        if apikey_env:
            curl_key = apikey_env if show_key else "<REDACTED>"
            print("curl (copy/paste):")
            print(f"curl '{_TWELVEDATA_ENDPOINT}' \\")
            print(f"  --get --data-urlencode 'symbol={state.ticker}' \\")
            print(f"  --data-urlencode 'interval={state.interval}' \\")
            print(f"  --data-urlencode 'outputsize={state.outputsize}' \\")
            print(f"  --data-urlencode 'format=JSON' \\")
            print(f"  --data-urlencode 'apikey={curl_key}'")
        else:
            print("Note: TWELVE_DATA_API_KEY not set in environment; URL shown without apikey.")
            print("      Set it via: export TWELVE_DATA_API_KEY='YOUR_KEY'")
