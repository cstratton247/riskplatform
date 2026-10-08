# riskplatform

Financial risk & predictive analytics platform. See `docs/milestone1_spec.md` for the
target definition, data dictionary, validation method, and leakage controls.

## Quickstart
```bash
cp .env.example .env          # edit the password
docker compose up -d          # starts Postgres and runs migrations/ on first init
pip install -e ".[dev]"
pytest
```
To re-run migrations from scratch: `docker compose down -v && docker compose up -d`.

## Layout
- `migrations/`  SQL schema and seed data (run in filename order)
- `src/riskplatform/`  library code; API and UI are thin layers over it
- `config/config.yaml`  all tunable parameters
- `tests/`  unit and leakage tests
- `docs/`  specification, model development document, validation report
