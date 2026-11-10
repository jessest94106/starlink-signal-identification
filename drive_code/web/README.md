# Husky Satellite Monitor

A self-hosted React + FastAPI satellite monitoring application built from [the design](husky_satellite_monitor_DESIGN.md). One workstation owns the shared receiver, tracking engine, authentication, configuration, history, and prediction data. No cloud services or external browser assets are used.

## Run on this workstation

Dependencies and the production frontend have been installed in this workspace. Create an operator, then start the local services:

```bash
.venv/bin/python scripts/create_user.py YOUR_USERNAME
.venv/bin/python scripts/run_local.py --mode simulator
```

Open **http://127.0.0.1:8000**. Guests can view the dashboard without logging in. Log in with the provisioned account, open **Receiver & processing**, and click **Start acquisition**. The simulator produces synthetic complex I/Q that passes through the supplied `TrackingEngine`'s actual FFT, CA-CFAR, association, and Kalman tracking. Its source is visibly labeled `simulator`.

Capture starts **stopped** at boot. Configuration changes stop acquisition, apply acknowledged service settings, create new capture/processing session IDs, increment the shared version, and broadcast to all viewers. Start acquisition again after applying settings. The operator form retains its original version until it closes or you reload active values, so concurrent edits return a visible conflict rather than overwrite newer settings.

For a fresh installation, install Python 3.12+, `python3-venv`, and Node 20+, then run:

```bash
bash scripts/bootstrap.sh
```

The virtual environment includes system site packages to access the workstation's installed UHD Python bindings. Python dependencies resolve inside the environment first. NumPy is pinned below 2 because the installed UHD 4.6 bindings were compiled against the NumPy 1.x ABI. `requirements.lock.txt` records the environment used for verification; `frontend/package-lock.json` pins the frontend installation.

## Sources

**Recorded SigMF replay** uses the supplied `SigMFReader` and real tracker:

```bash
.venv/bin/python scripts/run_local.py --mode sigmf --recording /path/to/capture.sigmf-data
```

The adjacent `.sigmf-meta` is required by the supervisor; its sample rate and center frequency initialize the shared receiver configuration before startup. Supported data: complex floating point `cf32_le`, `cf32_be`, `cf64_le`, `cf64_be`. Replay is paced at the recording epoch rate. Original recording UTC timestamps are preserved when present. End-of-file stops capture; Start begins a new replay/session. Raw files stay in Capture/Processing and are never available through HTTP.

**USRP acquisition** reuses the UHD setup from `capture_ofdm.py` in a modular continuous service. The original one-shot script is preserved and never imported, since it acquires hardware at import time.

Read-only discovery found two B210/NI2901 devices on this workstation: **31EEB59** and **34D9DCF**. The first device, **31EEB59**, was selected by the operator and is saved in `config/workstation.env`. The supervisor loads that file; environment variables can override it. To explicitly select a receiver:

```bash
HUSKY_USRP_ARGS='type=b200,serial=31EEB59' \
  .venv/bin/python scripts/run_local.py --mode usrp
```

Receiver center frequency is RF. Without an LNB, LO defaults to zero. For an LNB set `HUSKY_LNB_LO_HZ` (for example `9750000000`) and configure an RF center above the LO. `HUSKY_ANTENNA` defaults to `RX2`. Gain, sample rate, bandwidth, and frequency readback must match requested settings within tolerance; coerced values are rejected with an error that reports the supported value. Metadata is committed only after service confirmation. A disconnected receiver leaves web/health endpoints available; reset/reapply retries receiver discovery.

## Architecture

```text
Capture process -- bounded complex64 ZeroMQ --> Processing process
  UHD / simulator / SigMF                      supplied TrackingEngine
                                               precomputed prediction lookup
                                                        |
                                                  processed JSON only
                                                        v
React <---- WebSocket / REST ---- FastAPI ---- SQLite metadata + auth
                                   |
                          shared rolling history
```

Internal sockets bind to **127.0.0.1 only**: capture commands 5581, I/Q 5582, processing commands 5583, processed events 5584. The web backend sends commands through Processing to Capture; it never receives raw I/Q. Source and command boundaries permit a later C++/UHD acquisition replacement.

- `capture/service.py`: continuous capture, real UHD readback, health, bounded delivery.
- `processing/tracker_adapter.py`: thin adapter around the supplied engine; sends latest estimates rather than accumulated histories.
- `processing/prediction_store.py`, `satellite_matcher.py`: precomputed trajectory validation, lookup, interpolation, RMSE matching. See [prediction format](prediction_data/README.md).
- `backend/app/`: local Argon2 accounts, hashed server-side session tokens, cookie login, role checks, origin protection, login throttling, SQLite audit and session/config metadata, optimistic version locking, history, WebSocket fanout.
- `frontend/src/`: responsive dashboard with frequency overview/detail, strongest-detection SNR, Doppler frequency/rate, measured/predicted comparison, track filtering, sky trajectories, local pause, playback, browser-only display controls, and exponential reconnect with recent history recovery.

The supplied tracker is preserved with two additive optional methods, `prune_confirmed_history` and `skip_epochs`, for continuous streaming. It removes old histories and expired terminated tracks while preserving lifetime density, track identities, and Kalman state. `skip_epochs` advances the association clock across dropped I/Q blocks without fabricating measurements. Offline callers retain full history unless they invoke it.

## Storage and operation

SQLite (`data/monitor.db`) persists users, hashed login sessions, active configuration, configuration history, capture/processing session metadata, and the audit log. Authentication cookies are HttpOnly, SameSite=Lax, and expire after 12 hours. No public registration or default operator exists. Viewer accounts can be provisioned with `create_user.py --role viewer` and cannot issue commands.

Processed history is shared across viewers, indexed in events by processing session, retained for up to one hour, and additionally bounded by 1,000,000 events and a 2 GiB serialized-size estimate. These limits provide headroom for an hour at the workstation's observed capture rate; unusually dense detection traffic can still shorten the available interval. Browser history requests are capped and uniformly thin updates while preserving lifecycle/match metadata within the response limit. The frontend keeps at most 12,000 live epochs and limits plot drawing to about 1,500 points per series. Full processed history is intentionally in memory and does not survive a restart; disk persistence of large processed sessions is optional in the design and is not enabled. Raw recording persistence is handled by external SigMF captures.

I/Q queues hold 16 blocks. Slow processing drops new I/Q blocks rather than accumulate latency; sequence gaps are reported. WebSocket queues hold 256 events per viewer. Slow viewers receive a resync event and reconnect to recover recent history. The engine retains only a minute of confirmed history in streaming mode. Failures stop stale processing, preserve prior history, and surface health messages in the dashboard.

`/api/status` exposes queue depth/capacity, I/Q sequence gaps, capture drops, disconnected slow viewers, prediction errors, and p50/p95/p99/maximum latency summaries for processing, I/Q completion to processing, I/Q completion to WebSocket send, and client-reported receive-to-render timing. Replay timestamps and wall-clock latency timestamps are separate. Browser rendering timing is approximate and uses two animation frames, not a calibrated end-to-end clock measurement. Logs rotate at `data/monitor.log`, `data/capture.log`, and `data/processing.log` (5 MB each, three backups); audit events persist in SQLite.

## APIs

Public: `GET /api/status`, `/api/config/active`, `/api/history`, `/api/history/tracks/{track_id}`; `WS /ws/live`.

Auth: `POST /api/auth/login`, `/api/auth/logout`; `GET /api/auth/me`.

Operator: `PATCH /api/config/receiver`, `/api/config/processing`; `POST /api/capture/start`, `/api/capture/stop`, `/api/processing/reset`.

All operator mutations require `expected_config_version`. Receiver patches use flat fields. Processing patches use the existing configuration groups: `detector`, `tentative`, `kalman`, `manager`. Unknown keys, nonfinite values, invalid geometry, and unbounded track capacities are rejected. Status codes: 401 unauthenticated, 403 forbidden, 409 stale version, 422 invalid configuration, 503 unavailable/rejecting local service.

History supports `seconds` (up to 3600), ISO-8601 `start`/`end`, `processing_session_id`, `after_sequence`, and `limit` (up to 50000). Active configuration and every update carry session IDs and versions. Strongest-detection SNR is explicitly epoch-level, not per-track SNR.

## Development and verification

```bash
.venv/bin/python -m pytest -q
npm --prefix frontend run build
npm --prefix frontend run test:e2e
```

Browser tests require Chromium: `cd frontend && npx playwright install chromium`. They provision an isolated temporary database and fixture account, start all three processes on separate test ports, and exercise guest viewing, operator login/acquisition, multiple viewers, shared configuration updates, history, and mobile layout. Screenshots are written to `frontend/test-results/`. Production accounts are untouched.

Backend tests cover guest/viewer authorization, cookie/session lifecycle, configuration validation, service rejection, versions/conflicts, audit records, multi-viewer broadcasting, real multi-process ZeroMQ streaming, and history. Processing tests exercise real SigMF I/Q through FFT/CFAR/tracking, compare pruned vs full-history estimates, and verify precomputed satellite matching.

For frontend development, run the supervisor and `npm --prefix frontend run dev`. Vite proxies REST/WebSocket to localhost:8000. Set `HUSKY_PUBLIC_ORIGIN=http://127.0.0.1:5173` on the supervisor for authenticated commands from the development origin.

## Desktop boot and HTTPS

### Public access with sat-track

#### Reserved zrok address

This workstation is connected and owns **https://funlab.shares.zrok.io**.
The public launcher defaults to that reserved zrok name and URL. To start it:

```bash
.venv/bin/python scripts/run_public.py --mode simulator
```

Use `--mode usrp` for the receiver. Guests may need to click **Visit Share**
on zrok's first-visit notice; no browser add-ons are needed.

The official zrok2 client is installed at `.tools/zrok2`. Create a free account at
[myzrok.io](https://myzrok.io/), then connect this workstation using your account
token (input is hidden and isn't stored in shell history):

```bash
.venv/bin/python scripts/connect_zrok.py
.tools/zrok2 create name -n public funlab
.tools/zrok2 list names
```

If `funlab` is unavailable, reserve `funlab-sat-track` or another available name.
Copy the exact public URL printed by `list names`. Stop the existing public
launcher, then launch using that URL and the matching reserved name:

```bash
.venv/bin/python scripts/run_public.py --provider zrok --name funlab --public-url https://YOUR-RESERVED-HOSTNAME --mode simulator
```

The launcher sets the exact public origin and HTTPS-only cookies, and supervises
the tunnel and local services together. Reserved names survive across sessions;
the workstation and tunnel still need to stay running. Free unverified accounts
show an initial anti-phishing notice. No browser extensions are required.
See [reserved names](https://netfoundry.io/docs/zrok/how-tos/shares/manage-reserved-names/)
and [free-plan limits](https://zrok.io/pricing/).

#### Other tunnel providers

The `sat-track` public launcher can also expose the website through
a Cloudflare Quick Tunnel. No account, domain, browser add-ons,
or router port forwarding is required. With `cloudflared` installed, run:

```bash
.venv/bin/python scripts/run_public.py --provider cloudflare --mode simulator
```

Free addresses are assigned by the provider; the website displays `sat-track`.
The alternative `--provider localhost.run` uses the installed SSH client.
See [localhost.run documentation](https://localhost.run/docs/).
Its SSH host keys are stored in `data/sat-track-known-hosts`; changed host keys are rejected.

To install Cloudflare's official `cloudflared` binary on Linux x86-64:

```bash
mkdir -p .tools
curl -fL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o .tools/cloudflared
chmod +x .tools/cloudflared
```

Stop any existing `run_local.py` instance first, then run:

```bash
.venv/bin/python scripts/run_public.py --mode simulator
```

Use `--mode usrp` for hardware, or `--mode sigmf --recording /path/to/capture.sigmf-data`
for replay. The launcher prints the public HTTPS link and writes it to
`data/sat-track-url.txt`. Open that link from any internet-connected computer.
Guests can view the dashboard; acquisition and configuration still require operator
login. The launcher sets the exact public origin and HTTPS-only session cookies.
Acquisition starts stopped. Keep the workstation online and the launcher running;
Ctrl+C stops the tunnel and website. Tunnel diagnostics are in `data/sat-track-tunnel.log`.

Free tunnel URLs are temporary and can change; these tunnels do not provide a
production uptime guarantee. Cloudflare Quick Tunnels are intended for testing,
with a 200 concurrent request limit.
For a permanent address such as `sat-track.example.com`, configure a named
Cloudflare Tunnel using a domain you control, or use the Caddy setup below.
See [Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).

An example [systemd unit](config/husky-monitor.service.example) starts the supervised services on desktop boot. Review its paths, source mode, and receiver serial, then install it locally with your administrator privileges:

```bash
sudo cp config/husky-monitor.service.example /etc/systemd/system/husky-monitor.service
sudo systemctl daemon-reload
sudo systemctl enable --now husky-monitor
```

No system service is installed automatically. The launcher stops sibling processes if one exits; systemd restarts the group. Acquisition remains stopped on restart.

For remote access, run Caddy on the same workstation using [the example configuration](config/Caddyfile.example), set `HUSKY_PUBLIC_ORIGIN=https://YOUR_HOSTNAME` and `HUSKY_SECURE_COOKIES=1`, and keep Uvicorn on localhost. Only the reverse proxy should be remotely reachable. `--host`/`--port` customize the web listener. Internal ports never bind externally. The four `HUSKY_*` IPC endpoint environment variables allow isolated test instances.

Physical reception, RF calibration, real satellite predictions, and long-duration hardware stability must be verified with the selected receiver and the university's prediction data. Simulator and recorded-I/Q tests do not establish those hardware results.

## Verified on this workstation

The completed build passed 33 Python tests, the real-browser desktop/mobile test, and the frontend production build. A brief live check on B210 serial `31EEB59` acquired 500,000 complex samples at 500 kS/s and processed 100 epochs through the supplied tracker. No confirmed satellite signal was observed during that one-second hardware check. The hardware stream was stopped afterward. Long-duration hardware stability and real satellite identification remain dependent on the RF setup and precomputed prediction data.

### Download historical plot data

The dashboard toolbar offers **Export minutes** (0.1–60), **All retained history**, and **Download JSON**. The selected range ends at the latest retained event, independently of playback and display filters. Downloads include every retained event in the range, without playback thinning.

History is in memory and disappears on backend restart. Retention is bounded by one hour, 1,000,000 events, and an estimated 2 GiB, whichever fills first. Export range metadata reports the oldest available event and the actual exported range; previously evicted data cannot be recovered.

`GET /api/history/export?seconds=300` downloads the last five minutes; omit `seconds` for all retained history. The JSON contains `schema_version`, `range`, `storage`, `feature_definitions`, `configurations`, and `events`. Events reference their original settings through `configuration_ref` / `config_version`, and preserve capture and processing session IDs. Each plotted feature includes its name, value, units, plot IDs, timestamp in Unix seconds, and track ID. Track IDs are scoped to a processing session. Epoch-wide strongest-detection SNR has a null track ID. Satellite-match events preserve prediction trajectories and identify their frequency plots.

New capture settings include receiver mode, actual hardware serial, device information, antenna, master clock rate, LNB LO, channel and sample formats, alongside sample rate, RF center frequency, bandwidth, gain and processing settings. Older records without receiver identity remain unknown rather than using the currently connected radio. Capture/processing services must load the updated code before new hardware metadata is recorded. Exports contain measurements and track events, not raw I/Q recordings.

See `docs/history-export-example.json` for an illustrative complete export.
