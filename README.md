# Gauteng Multi-Modal Commuter Copilot

An AI commuter assistant for Johannesburg and Pretoria that plans trips combining **Rea Vaya BRT** with the **Gautrain**. Ask in plain English, for example *"How do I get from Dobsonville to Sandton during peak hours?"*, and it returns the legs, the total 2026 fare, travel time, card rules and transfer tips.

Built for **Google DevFest Johannesburg 2026 — The Vibe Lounge (Developer Track)**.

## The problem

Going from Soweto to Sandton or Pretoria means combining Rea Vaya feeder and trunk routes, a transfer at Park Station, two card systems with different minimum balances, and separate peak and off-peak tariffs. Commuters often overpay, tap out in the wrong place, or reach the barrier without enough on their card.

## How it works

```
User question
     │
     ▼
Input guardrails (Pydantic: length, type, control characters)
     │
     ▼
calculate_commute_plan()  ← deterministic Python, routes_2026.json tariffs
     │
     ├──────────────────────────────┐   (run in parallel)
     ▼                              ▼
Gemini: plan summary          Gemini + Google Search grounding:
(system instruction,          live service disruption check
 verified tool data only)     (4 s timeout, graceful fallback)
     │                              │
     ▼                              │
Output guardrail: every Rand        │
amount must be a verified fare      │
     │                              │
     └──────────────┬───────────────┘
                    ▼
     FastAPI JSON → web UI (Markdown sanitised with DOMPurify)
```

- **Python calculates every fare, never the model.** `calculate_commute_plan` looks up each leg in `routes_2026.json`: Rea Vaya distance bands and Gautrain station-pair fares, for peak and off-peak.
- **Gemini only writes the summary.** It turns the verified numbers into a short plan with card and transfer rules.
- **Live alerts.** A separate Gemini call with `google_search` grounding checks for current disruptions and lists its sources.
- **Model selection.** At startup the app sends a small test request to each model in `CANDIDATE_MODELS` and uses the first one that responds.

## Supported corridors

| Corridor | Modes |
|---|---|
| Dobsonville → Sandton | Rea Vaya F2 feeder + T1 trunk → Gautrain (Park Station → Sandton) |
| Thokoza Park (Soweto) → Rosebank | Rea Vaya T1 trunk → Gautrain (Park Station → Rosebank) |
| Maponya Mall (Klipspruit) → Hatfield | Rea Vaya T1 trunk → Gautrain (Park Station → Hatfield) |

To add a corridor, add an entry to `corridors` in `routes_2026.json`. No code changes are needed.

## Security and privacy

| Area | What we do |
|---|---|
| Input validation | The prompt must be 3–300 printable characters. Malformed JSON and wrong types return a clear `422` instead of an error page. Tool arguments are checked against a typed `PlanArgs` schema. |
| Fare accuracy | Every Rand amount in the model's answer is checked against the tariff dataset. If any amount isn't in the data, the answer is replaced with a summary built directly from the data. |
| Prompt injection | User text is wrapped in `<user_request>` tags, and the system instruction tells the model to treat it as untrusted. Only the matched route name is sent to Google Search, never the raw question. |
| XSS | Model output is rendered with `marked` and sanitised with `DOMPurify`. Source links are HTML-escaped. |
| API key | Read only from the `GEMINI_API_KEY` environment variable on the server. `.env` is gitignored. The key never reaches the browser. |
| POPIA | No database, no cookies, and access logging is off (`--no-access-log`). The UI tells people not to enter personal details and that their question goes to Google Gemini for processing. |

## Run locally

Requires Python 3.10+ and a [Gemini API key](https://aistudio.google.com/apikey).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then add your key to .env
set -a && source .env && set +a

uvicorn app:app --host 0.0.0.0 --port 8000 --no-access-log
```

Open http://localhost:8000.

To share it publicly:

```bash
npx cloudflared tunnel --url http://localhost:8000
```

> **Note:** Google Search grounding needs a Gemini API plan that includes grounding quota. Without it, the live-alerts card shows "live search unavailable", and the rest of the app works normally.

## API

`POST /api/plan`

```bash
curl -X POST http://localhost:8000/api/plan \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Best off-peak route from Thokoza Park to Rosebank?"}'
```

The response contains:

- `plan`: a Markdown summary.
- `tool_data`: the verified fares, legs and transfer notes.
- `live_alerts`: the result of the live disruption check.
- `grounding_sources`: the web sources behind the live alerts.

Invalid input returns `422` with `{"error": "..."}`.

## Project structure

```
app.py              FastAPI app: guardrails, fare tool, Gemini calls, web UI
routes_2026.json    2026 Rea Vaya distance bands, Gautrain fares, corridors
requirements.txt    Pinned dependencies
.env.example        Template for GEMINI_API_KEY
```

## Disclaimer

The fares in `routes_2026.json` are a demo dataset for DevFest JHB 2026. Before you travel, check current fares with [Rea Vaya](https://www.reavaya.org.za) and [Gautrain](https://www.gautrain.co.za).
