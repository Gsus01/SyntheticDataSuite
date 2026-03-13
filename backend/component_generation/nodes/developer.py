from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from component_generation.context import load_prompt
from component_generation.llm import LLMClient
from component_generation.llm_trace import write_llm_request, write_llm_response
from component_generation.schemas import GeneratedComponentFiles
from component_generation.state import PipelineState
from component_generation.validation import (
    validate_generated_component_plan,
    validate_generated_componentspec,
)
from component_spec import ComponentSpec

logger = logging.getLogger(__name__)

DEFAULT_VERSION = "v0.1.0"
DEFAULT_IMAGE_PREFIX = "sds"


def _sanitize_kebab(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\-]+", "-", value.strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "component"


def _normalize_outputs(outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for port in outputs:
        if not isinstance(port, dict):
            continue
        name = port.get("name") or "output-data"
        path = port.get("path") or "/data/outputs/output"
        role = port.get("role") or "data"
        normalized.append({"name": name, "path": path, "role": role})
    return normalized


def _render_main_py(component: Dict[str, Any]) -> str:
    outputs = _normalize_outputs(component.get("outputs") or [])
    output_targets = []
    for port in outputs:
        path = port.get("path", "")
        output_targets.append(path)

    output_targets_literal = json.dumps(output_targets, indent=2)
    return "\n".join(
        [
            '"""Auto-generated stub component."""',
            "from __future__ import annotations",
            "",
            "import argparse",
            "from pathlib import Path",
            "",
            'DEFAULT_INPUT_DIR = "/data/inputs"',
            'DEFAULT_OUTPUT_DIR = "/data/outputs"',
            'DEFAULT_CONFIG_DIR = "/data/config"',
            "",
            "OUTPUT_TARGETS = " + output_targets_literal,
            "",
            "",
            "def _resolve_output_path(output_dir: Path, target: str) -> Path:",
            "    if not target:",
            '        return output_dir / "output.txt"',
            '    prefix = "/data/outputs"',
            "    if target.startswith(prefix):",
            "        rel = target[len(prefix):].lstrip('/')",
            "        if not rel:",
            '            return output_dir / "output.txt"',
            "        return output_dir / rel",
            "    return output_dir / Path(target).name",
            "",
            "",
            "def main() -> None:",
            '    parser = argparse.ArgumentParser(description="Synthetic Data Suite component")',
            '    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)',
            '    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)',
            '    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)',
            "    args = parser.parse_args()",
            "",
            "    input_dir = Path(args.input_dir)",
            "    output_dir = Path(args.output_dir)",
            "    output_dir.mkdir(parents=True, exist_ok=True)",
            "",
            "    # Placeholder: list inputs and emit stub outputs",
            "    inputs = list(input_dir.glob('*')) if input_dir.exists() else []",
            "    for target in OUTPUT_TARGETS or ['']:",
            "        output_path = _resolve_output_path(output_dir, target)",
            "        output_path.parent.mkdir(parents=True, exist_ok=True)",
            "        output_path.write_text(",
            '            "Generated placeholder output. Inputs found: " + str(len(inputs)) + "\\n",',
            "            encoding='utf-8',",
            "        )",
            "",
            "",
            "if __name__ == \"__main__\":",
            "    main()",
            "",
        ]
    )


def _render_dockerfile() -> str:
    return "\n".join(
        [
            "FROM python:3.12-slim",
            "",
            "WORKDIR /app",
            "COPY requirements.txt .",
            "RUN pip install --no-cache-dir -r requirements.txt",
            "COPY . .",
            "",
            'CMD ["python","main.py"]',
            "",
        ]
    )


def _render_componentspec(component: Dict[str, Any], name: str) -> Dict[str, Any]:
    component_type = component.get("type") or "other"
    title = component.get("title") or name.replace("-", " ").title()
    description = component.get("description") or ""
    inputs = component.get("inputs") or []
    outputs = component.get("outputs") or []
    parameters = component.get("parameters_defaults") or {}

    spec: Dict[str, Any] = {
        "apiVersion": "sds/v1",
        "kind": "Component",
        "metadata": {
            "name": name,
            "version": DEFAULT_VERSION,
            "type": component_type,
            "title": title,
            "description": description,
        },
        "io": {
            "inputs": inputs,
            "outputs": outputs,
        },
        "runtime": {
            "image": f"{DEFAULT_IMAGE_PREFIX}/{name}:dev",
        },
    }

    if parameters:
        spec["parameters"] = {"defaults": parameters}

    return spec


def _truncate(text: str, max_chars: int = 80_000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]\n"


def node_developer(state: PipelineState) -> Dict[str, Any]:
    plan = state.get("plan") or {}
    components = plan.get("components") or []
    out_dir = Path(state.get("out_dir") or "components/generated-sessions").resolve()
    session_dir = state.get("session_dir")
    llm: LLMClient | None = state.get("llm")
    context = state.get("developer_context") or ""
    source = _truncate(state.get("combined_source") or "")
    disable_llm = bool(state.get("disable_llm"))
    use_structured_output = bool(state.get("structured_output", True))
    out_dir.mkdir(parents=True, exist_ok=True)
    session_path = Path(session_dir).resolve() if session_dir else None
    components_root = (session_path / "components") if session_path else out_dir

    logger.info("developer: generating %s component(s)", len(components))

    generated_index: Dict[str, Dict[str, str]] = {}

    for comp in components:
        validate_generated_component_plan(comp)
        raw_name = comp.get("name") or "component"
        name = _sanitize_kebab(raw_name)
        comp_type = comp.get("type") or "other"
        folder = components_root / comp_type / name
        folder.mkdir(parents=True, exist_ok=True)

        main_path = folder / "main.py"
        docker_path = folder / "Dockerfile"
        spec_path = folder / "ComponentSpec.json"
        reqs_path = folder / "requirements.txt"
        readme_path = folder / "README.md"

        if llm and not disable_llm:
            system = load_prompt("developer_system.md")
            schema_hint = ""
            if not use_structured_output:
                schema_hint = (
                    "\n\nOUTPUT JSON SCHEMA:\n"
                    + json.dumps(
                        GeneratedComponentFiles.model_json_schema(),
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            user = (
                "COMPONENT PLAN:\n"
                + json.dumps(comp, indent=2, ensure_ascii=False)
                + "\n\nDEVELOPER CONTEXT:\n"
                + context
                + "\n\nSOURCE (may be truncated):\n"
                + source
            )
            if schema_hint:
                user += schema_hint
            try:
                format_schema = (
                    GeneratedComponentFiles.model_json_schema()
                    if use_structured_output
                    else None
                )
                write_llm_request(
                    state.get("session_dir"),
                    "developer",
                    name,
                    {
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "format_schema": format_schema,
                        "structured_output": use_structured_output,
                        "provider": state.get("llm_provider"),
                        "model": state.get("llm_model"),
                    },
                )
                raw = llm.invoke(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    format_schema=format_schema,
                )
                write_llm_response(state.get("session_dir"), "developer", name, raw)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("developer: LLM raw response for %s:\n%s", name, raw.strip())
                files = GeneratedComponentFiles.model_validate_json(raw)
                try:
                    spec = ComponentSpec.model_validate(files.componentspec)
                    validate_generated_componentspec(spec, component_name=name)
                except Exception as exc:
                    logger.error(
                        "developer: invalid ComponentSpec for %s: %s", name, exc
                    )
                    raise RuntimeError(
                        f"Developer LLM returned invalid ComponentSpec for {name}."
                    ) from exc
                main_path.write_text(files.main_py, encoding="utf-8")
                docker_path.write_text(files.dockerfile, encoding="utf-8")
                spec_path.write_text(
                    json.dumps(files.componentspec, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                reqs_path.write_text(
                    (files.requirements_txt or "").strip() + "\n", encoding="utf-8"
                )
                if (files.readme_md or "").strip():
                    readme_path.write_text(files.readme_md, encoding="utf-8")
                else:
                    readme_path.write_text(
                        f"# {name}\n\nAuto-generated component stub.\n",
                        encoding="utf-8",
                    )
            except Exception as exc:
                logger.error("developer: LLM failed for %s: %s", name, exc)
                raise RuntimeError(f"Developer LLM failed for {name}.") from exc
        else:
            if not disable_llm:
                raise RuntimeError(
                    "LLM not configured; rerun with --no-llm to skip LLM."
                )
            main_path.write_text(_render_main_py(comp), encoding="utf-8")
            docker_path.write_text(_render_dockerfile(), encoding="utf-8")
            spec_payload = _render_componentspec(comp, name)
            validate_generated_componentspec(
                ComponentSpec.model_validate(spec_payload),
                component_name=name,
            )
            spec_path.write_text(
                json.dumps(spec_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            reqs_path.write_text("\n", encoding="utf-8")
            readme_path.write_text(
                f"# {name}\n\nAuto-generated component stub.\n",
                encoding="utf-8",
            )

        generated_index[name] = {
            "folder": str(folder),
            "main.py": str(main_path),
            "Dockerfile": str(docker_path),
            "ComponentSpec.json": str(spec_path),
            "requirements.txt": str(reqs_path),
            "README.md": str(readme_path),
        }

        logger.info("developer: generated %s (%s)", name, folder)

    if session_path:
        session_path.mkdir(parents=True, exist_ok=True)
        (session_path / "generated_index.json").write_text(
            json.dumps(generated_index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        _write_session_scripts(session_path, generated_index)

    return {"generated_index": generated_index}


def _write_session_scripts(session_path: Path, generated_index: Dict[str, Dict[str, str]]) -> None:
    build_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
        "COMPONENTS_DIR=\"$SCRIPT_DIR/components\"",
        "",
        "eval \"$(minikube docker-env)\"",
        "",
    ]

    register_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
        "COMPONENTS_DIR=\"$SCRIPT_DIR/components\"",
        "BACKEND_URL=\"${BACKEND_URL:-http://localhost:8000}\"",
        "",
    ]

    components_root = session_path / "components"

    for meta in generated_index.values():
        folder = Path(meta["folder"])
        try:
            rel = folder.relative_to(components_root)
        except ValueError:
            rel = folder
        rel_str = rel.as_posix()

        spec_path = folder / "ComponentSpec.json"
        image = ""
        if spec_path.exists():
            try:
                payload = json.loads(spec_path.read_text(encoding="utf-8"))
                runtime = payload.get("runtime") or {}
                image = runtime.get("image") or ""
            except Exception:
                image = ""

        if image:
            build_lines.append(f"docker build -t {image} \"$COMPONENTS_DIR/{rel_str}\"")
        else:
            build_lines.append(f"docker build -t <set-image> \"$COMPONENTS_DIR/{rel_str}\"")

        register_lines.extend(
            [
                f"python3 - \"$COMPONENTS_DIR/{rel_str}/ComponentSpec.json\" <<'PY' | curl -s -X POST \"$BACKEND_URL/components\" -H 'Content-Type: application/json' -d @-",
                "import json,sys",
                "path = sys.argv[1]",
                "with open(path, 'r', encoding='utf-8') as f:",
                "    spec = json.load(f)",
                "print(json.dumps({'spec': spec, 'activate': True}))",
                "PY",
                "",
            ]
        )

    build_script = "\n".join(build_lines) + "\n"
    register_script = "\n".join(register_lines) + "\n"

    build_path = session_path / "build_images.sh"
    register_path = session_path / "register_components.sh"

    build_path.write_text(build_script, encoding="utf-8")
    register_path.write_text(register_script, encoding="utf-8")

    try:
        build_path.chmod(0o755)
        register_path.chmod(0o755)
    except Exception:
        pass
