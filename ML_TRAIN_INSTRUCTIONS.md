# ML Train / Server — Documentation & What To Do

Ito ay isang praktikal na gabay para sa `ml` training server na kasama sa proyekto (Flask + scikit-learn). Nakalagay dito kung ano ang dapat gawin kapag magta-troubleshoot, paano mag-run locally, at kung paano i-integrate sa mobile client.

## Buod ng Layunin
- Ang server ay nag-eexpose ng endpoints para sa clustering (`kmeans`) at rekomendasyon (`knn`).
- Ang mobile client tumatawag sa endpoint na karaniwang `POST /ml` gamit ang JSON payload.
- Dapat mag-return ang server ng `application/json` responses; kung HTML (404/maintenance) ang natatanggap ng client, magdudulot iyon ng JSON parse errors.

## Mahahalagang File / Function
- `ml_model.train_kmeans(...)` — nagtra-train o nag-aapply ng tertiles depende sa dataset size.
- `ml_model.knn_recommend(...)` — nagbibigay ng pinakamalapit na property rekomendasyon kasama confidence at distance.
- `preprocess.preprocess_data(...)` — naglilinis at nagsi-scale ng data (RobustScaler), nag-`remove_outliers` at nagbabalik ng scaled array + scaler + cleaned DataFrame.
- Flask API routes: `/` (docs), `/check-db`, `/ml`.

## Environment / Dependencies
1. Gumawa ng virtual environment at i-install ang requirements:

```bash
python -m venv .venv
source .venv/bin/activate   # Unix/macOS
.\\.venv\\Scripts\\activate  # Windows PowerShell
pip install -r requirements.txt
```

2. Siguraduhing naka-set ang `MONGO_URI` sa `.env` o environment variables at na-access ang database.

## How to Run Locally

```bash
export FLASK_APP=app.py    # o kung ibang filename
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000
```
o (Windows PowerShell)
```powershell
$env:FLASK_APP = 'app.py'
$env:FLASK_ENV = 'development'
flask run --host=0.0.0.0 --port=5000
```

## Test Endpoints (examples)
- Check server root:
```bash
curl -i https://your-ml-host.example.com/
```
- POST `/ml` for KNN recommendations (example payload):
```bash
curl -i -X POST https://your-ml-host.example.com/ml \
  -H "Content-Type: application/json" \
  -d '{"mode":"knn","price":1000,"latitude":14.5995,"longitude":120.9842,"k":5}'
```
- POST `/ml` for KMeans clustering (example payload):
```bash
curl -i -X POST https://your-ml-host.example.com/ml \
  -H "Content-Type: application/json" \
  -d '{"mode":"kmeans","n_clusters":3}'
```

Important: i-check ang HTTP status at `Content-Type` header — dapat `application/json` at status `200` para normal flow.

## Common Issues & Fixes

- 404 Not Found (HTML response)
  - Sanhi: Flask app hindi naka-deploy sa expected path (e.g., ibang route mount), o custom domain/direct ingress na nagre-serve ibang service.
  - Gawin:
    1. I-check ang server logs kung may incoming request at anong path ang tinatanggap.
    2. I-test ang root gamit `curl` at `curl -i` para makita headers at body.
    3. Kung app mounted sa subpath (hal. `/api`), i-update ang mobile `ML_API_URL` o gumawa ng route shim sa server (redirect o alias).

- JSON Parse Error (Unexpected character '<')
  - Nangyayari kapag body ay HTML (404 page). Solusyon: tiyakin ang `Content-Type: application/json` at i-handle ang error sa server side.

- Database connection errors
  - I-verify ang `MONGO_URI` at connectivity (ip, port, auth). Gamitin ang `check-db` endpoint para mabilis na sanity check.

## Recommended Server-Side Improvements
- Return consistent error objects: `{ error: 'message', code: ..., details: ... }` at proper status codes.
- Add a lightweight `/health` or `/ping` endpoint na nagre-return ng JSON para sa readiness checks.
- Enable structured logging (timestamp + level + request id) para madaling i-trace requests mula sa mobile client.
- Ensure CORS naka-enable (Flask-CORS) at naka-configure para sa iyong mobile client origin.

## Notes on ML Logic
- For small datasets (< ~30 samples) the server currently falls back to tertiles to avoid unstable clustering.
- `train_kmeans` returns `metrics` (silhouette, davies_bouldin, calinski_harabasz) when computable — gamitin ito para monitoring clustering quality.
- `preprocess_data` uses `RobustScaler` and caches the scaler in-memory (`scaler_cache`) to apply consistent scaling to query points.

## Mobile Client Integration Tips
- Mobile client should point `ML_API_URL` to the server base (e.g. `https://new-train-ml.onrender.com`). Prefer to expose `/ml` on the server (root path) to match client expectations.
- The client now attempts multiple candidate paths when `/ml` returns 404 — but best practice ay i-fix server so `/ml` works and returns JSON.

## Debugging Checklist (quick)
1. From your machine:
   - `curl -i https://new-train-ml.onrender.com/` — check 200 and JSON body
   - `curl -i -X POST https://new-train-ml.onrender.com/ml -H "Content-Type: application/json" -d '{"mode":"knn","price":1000,"latitude":14.6,"longitude":120.98}'`
2. If you get HTML 404, inspect server/router configuration and Render/hosting logs.
3. Check Flask logs for stack traces and exceptions (look for `request.get_json()` errors, or DB errors).
4. Ensure `MONGO_URI` is valid and DB has property records.

## Deployment / Production Notes
- Serve the Flask app behind a reverse proxy (nginx) or platform routing that forwards `/ml` to the Flask process.
- Configure health checks (use `/` or `/health`) so the host platform knows the service is up.
- Use environment variables for secrets and set `FLASK_ENV=production` on production.

---
Kung gusto mo, gagawin ko rin ang isang maliit na endpoint sa client na magpapakita ng verbose diagnostics (status ng ML server attempts) sa isang modal para mas mabilis makita ang issue. Sabihin mo lang kung gusto mo i-implement iyon.
