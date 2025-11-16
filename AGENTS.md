# Repository Guidelines

## Project Structure & Module Organization
The workspace is split between `backend/` (FastAPI services plus workflow orchestration helpers), `frontend/` (Next 15 + React Flow canvas), and `components/` (data-processing utilities with pytest suites in `components/preprocessing/tests/{unit,integration,blackbox}`). Kubernetes and MinIO manifests live in `deploy/`, reusable configs in `config/`, and developer scripts inside `scripts/dev/` drive Minikube, port-forwarding, and stack startup. Read `docs/dev-workflow.md` before wiring new services so manifests, helpers, and documentation land in the matching folder.

## Build, Test, and Development Commands
Use `make dev` for the whole stack (MinIO, Argo, backend on :8000, frontend on :3000). Targeted helpers include `make backend`, `make frontend`, `make k8s-up`, `make k8s-down`, plus `make port-forward` / `make port-forward-stop`. Backend dependencies are handled with uv:
```bash
uv sync --project backend
uv run --project backend uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Frontend work stays inside `frontend/` via `npm run dev`, `npm run build`, and `npm run lint`. Use `make workflow` only after Argo is reachable.

## Coding Style & Naming Conventions
Python modules follow 4-space indentation, type-annotated FastAPI endpoints, and descriptive module names (`workflow_builder.py`, `minio_helper.py`). Keep config objects in SCREAMING_SNAKE_CASE and prefer PascalCase for Pydantic models. TypeScript/React code follows the ESLint + Next defaults: camelCase for hooks/utilities, PascalCase for components, and colocated styling via Tailwind 4 tokens. Run `npm run lint` before pushing to keep imports ordered and hooks validated.

## Testing Guidelines
Pytest backs the preprocessing suite; mirror the folder layout (`unit`, `integration`, `blackbox`) when adding coverage and prefer descriptive `test_<behavior>` functions. Execute tests inside uv to reuse the locked environment:
```bash
uv run --project backend pytest components/preprocessing/tests
```
Front-end logic without tests should at least ship ESLint coverage; if you introduce reusable utilities, add vitest/jest coverage before merging.

## Commit & Pull Request Guidelines
Git history favors concise, present-tense summaries (`Enhance FlowEditor...`, `Add ArgoClient...`). Keep subject lines under ~72 chars, expand reasoning in the body if needed, and reference issue IDs. Pull requests should list the affected modules, manual test notes (commands run, clusters touched), and screenshots/GIFs for UI-facing changes, plus mention any new env vars (`MINIO_*`, `ARGO_*`). Request review only after `make dev` boots cleanly.

## Security & Configuration Tips
Never commit secrets—use `.env` overrides and export `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, and `ARGO_SERVER_*` locally. When forwarding ports, prefer the scripts in `scripts/dev/` so cleanup happens automatically; ad-hoc steps leave background processes that interfere with `make dev`.
