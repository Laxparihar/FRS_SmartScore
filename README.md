# FRS SmartScore

**FRS SmartScore** checks whether two photos show the same person. It is designed for identity
verification, comparing a person's selfie with the photo on their Aadhaar card, PAN card or Passport.

## Features
- Simple web page: upload two images and get the result
- REST API (`POST /api/verify`) that other systems can call
- 0-100 FRS score with MATCH / REVIEW / NO MATCH decision
- Handles sideways or upside-down ID card photos automatically
- Enlarges small, low-quality ID card faces before matching
- Image quality warnings (blurry, dark, small face, multiple faces)
- Runs fully offline on CPU, so no images are sent to any cloud service

## Run
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open http://localhost:8000. The first start downloads the face model (~280 MB) into `models/`.

## API
`POST /api/verify` (multipart form): `image1`, `image2`

```bash
curl -F image1=@selfie.jpg -F image2=@aadhaar.jpg http://localhost:8000/api/verify
```
Response: `frs_score` (0-100), `decision` (`MATCH` / `REVIEW` / `NO_MATCH`), `cosine_similarity`,
per-image face details and quality warnings. Swagger docs: `/docs`.

## Scoring
| Cosine similarity | FRS score | Decision |
|---|---|---|
| >= 0.35 | >= 70 | MATCH |
| 0.25 - 0.35 | 50 - 70 | REVIEW |
| < 0.25 | < 50 | NO_MATCH |

Thresholds can be changed with env vars `FRS_MATCH_THRESHOLD` and `FRS_REVIEW_THRESHOLD`.
Tune them on a set of your own real selfie-vs-ID pairs before go-live.

## Licence note
InsightFace `buffalo_l` model weights are for non-commercial use only. Get a commercial licence
or swap the model before production use.
# FRS_SmartScore
