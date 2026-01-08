# Repository Guidelines

## Project Structure & Module Organization
- `backend/`: FastAPI API plus workflow/orchestration helpers (Argo + MinIO integration).
- `frontend/`: Next.js 15 app with the React Flow canvas UI.
- `components/`: reusable, container-friendly data-processing code; preprocessing tests live in `components/preprocessing/tests/{unit,integration,blackbox}`.
- `deploy/`: Kubernetes manifests (stack + workflows).
- `scripts/dev/`: dev scripts used by `make` targets; prefer these over ad-hoc kubectl commands.
- `docs/`: contributor docs like `docs/dev-workflow.md` and `docs/new-component.md`.

## Build, Test, and Development Commands
- `make dev`: start the full local stack (MinIO, Argo, backend on `:8000`, frontend on `:3000`).
- `make backend`: install/sync backend deps with uv and run `uvicorn` in reload mode.
- `make frontend`: run the frontend dev server (`NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`).
- `make k8s-up` / `make k8s-down`: bring MinIO/Argo up/down in Minikube (`make k8s-up-minio`, `make k8s-up-argo` for targeted).
- `make port-forward` / `make port-forward-stop`: manage MinIO/Argo port-forwarding via `scripts/dev/port-forward.sh`.
- `make workflow`: submit `deploy/general_workflow.yaml` to Argo (requires Argo reachable).

Backend (uv) quickstart:
```bash
uv sync --project backend
uv run --project backend uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Coding Style & Naming Conventions
- Python: 4-space indentation, type-annotated FastAPI endpoints, descriptive module names (e.g., `workflow_builder.py`); Pydantic models in PascalCase; constants in SCREAMING_SNAKE_CASE.
- TypeScript/React: follow ESLint + Next defaults; components in PascalCase, hooks/utilities in camelCase; keep styling colocated (Tailwind).

## Testing Guidelines
- Use `pytest` for component tests; mirror the `unit/`, `integration/`, `blackbox/` layout and prefer `test_<behavior>` function names.
- Run: `uv run --project backend pytest components/preprocessing/tests`
- Frontend: run `npm run lint` in `frontend/` and add unit tests (vitest/jest) when introducing reusable utilities.

## Commit & Pull Request Guidelines
- Prefer imperative, present-tense subjects; conventional prefixes like `feat:`/`fix:` are common. Keep the subject under ~72 chars.
- PRs should describe the affected modules, verification steps (commands run / clusters touched), and include screenshots/GIFs for UI changes.
- Call out new env vars or config requirements (e.g., `MINIO_*`, `ARGO_*`).

## Security & Configuration Tips
- Never commit secrets—use local `.env` overrides and export `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, and `ARGO_SERVER_*`.
- Component containers should follow the I/O convention: `/data/inputs`, `/data/config`, `/data/outputs`.
