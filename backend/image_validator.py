"""Validate that Docker images exist in minikube's Docker daemon."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)

_TEMPLATE_REGISTRY_PATH = Path(__file__).resolve().parent / "workflow-templates.yaml"


def _should_skip_validation() -> bool:
    """Check if image validation should be skipped.

    Set SKIP_IMAGE_VALIDATION=true when running in a container where
    minikube/docker commands are not available.
    """
    value = os.getenv("SKIP_IMAGE_VALIDATION", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


@dataclass
class ImageValidationResult:
    """Result of validating Docker images."""

    all_present: bool
    missing_images: List[str]
    present_images: List[str]
    error: Optional[str] = None


def _get_minikube_docker_env() -> Dict[str, str]:
    """Get the Docker environment variables for minikube."""
    try:
        result = subprocess.run(
            ["minikube", "docker-env", "--shell=bash"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.warning("Failed to get minikube docker-env: %s", result.stderr)
            return {}

        env = os.environ.copy()
        for line in result.stdout.splitlines():
            if line.startswith("export "):
                # export DOCKER_HOST="tcp://192.168.49.2:2376"
                parts = line[7:].split("=", 1)
                if len(parts) == 2:
                    key = parts[0]
                    value = parts[1].strip('"')
                    env[key] = value
        return env
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("Could not get minikube docker-env: %s", e)
        return {}


def _list_minikube_images() -> Set[str]:
    """List all Docker images available in minikube's Docker daemon."""
    env = _get_minikube_docker_env()
    if not env:
        # Fallback: try using minikube ssh
        return _list_minikube_images_via_ssh()

    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        if result.returncode != 0:
            logger.warning("Failed to list docker images: %s", result.stderr)
            return set()

        images = set()
        for line in result.stdout.strip().splitlines():
            if line and line != "<none>:<none>":
                images.add(line)
                # Also add without docker.io/library/ prefix for matching
                if line.startswith("docker.io/library/"):
                    images.add(line[len("docker.io/library/") :])
                elif (
                    not line.startswith("docker.io/") and "/" not in line.split(":")[0]
                ):
                    # Add with prefix for matching
                    images.add(f"docker.io/library/{line}")
        return images
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("Could not list docker images: %s", e)
        return set()


def _list_minikube_images_via_ssh() -> Set[str]:
    """List Docker images via minikube ssh as fallback."""
    try:
        result = subprocess.run(
            ["minikube", "ssh", "docker images --format '{{.Repository}}:{{.Tag}}'"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("Failed to list images via minikube ssh: %s", result.stderr)
            return set()

        images = set()
        for line in result.stdout.strip().splitlines():
            line = line.strip().strip("'")
            if line and line != "<none>:<none>":
                images.add(line)
                if line.startswith("docker.io/library/"):
                    images.add(line[len("docker.io/library/") :])
                elif (
                    not line.startswith("docker.io/") and "/" not in line.split(":")[0]
                ):
                    images.add(f"docker.io/library/{line}")
        return images
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("Could not list images via minikube ssh: %s", e)
        return set()


def get_all_template_images() -> Dict[str, str]:
    """Get all images defined in workflow-templates.yaml.

    Returns:
        Dict mapping template name to image name.
    """
    if not _TEMPLATE_REGISTRY_PATH.exists():
        return {}

    with _TEMPLATE_REGISTRY_PATH.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    templates = raw.get("templates") or {}
    result = {}
    for name, body in templates.items():
        container = body.get("container") or {}
        image = container.get("image")
        if image:
            result[name] = image
    return result


def validate_images(template_names: List[str]) -> ImageValidationResult:
    """Validate that the images for the given templates exist in minikube.

    Args:
        template_names: List of template names to validate.

    Returns:
        ImageValidationResult with validation details.

    Note:
        Set SKIP_IMAGE_VALIDATION=true to skip validation when running in a
        container where minikube/docker commands are not available.
    """
    # Skip validation if configured (useful when running in a container)
    if _should_skip_validation():
        logger.info("Image validation skipped (SKIP_IMAGE_VALIDATION=true)")
        all_images = get_all_template_images()
        required = [all_images[name] for name in template_names if name in all_images]
        return ImageValidationResult(
            all_present=True,
            missing_images=[],
            present_images=required,
        )

    all_images = get_all_template_images()
    available_images = _list_minikube_images()

    if not available_images:
        return ImageValidationResult(
            all_present=False,
            missing_images=[],
            present_images=[],
            error="No se pudo conectar con el Docker de minikube. Asegúrate de que minikube está corriendo.",
        )

    required_images: Dict[str, str] = {}
    for name in template_names:
        if name in all_images:
            required_images[name] = all_images[name]

    missing = []
    present = []

    for template_name, image in required_images.items():
        # Check both with and without docker.io/library prefix
        image_variants = {image}
        if image.startswith("docker.io/library/"):
            image_variants.add(image[len("docker.io/library/") :])
        else:
            image_variants.add(f"docker.io/library/{image}")

        if any(v in available_images for v in image_variants):
            present.append(image)
        else:
            missing.append(image)

    return ImageValidationResult(
        all_present=len(missing) == 0,
        missing_images=missing,
        present_images=present,
    )


def validate_all_images() -> ImageValidationResult:
    """Validate all images defined in workflow-templates.yaml."""
    all_images = get_all_template_images()
    return validate_images(list(all_images.keys()))


def get_build_command_for_image(image: str) -> Optional[str]:
    """Get the build command for a missing image.

    Returns the command to build the image, or None if unknown.
    """
    # Extract the image name without registry prefix
    name = image
    if name.startswith("docker.io/library/"):
        name = name[len("docker.io/library/") :]

    # Remove :latest or other tags
    if ":" in name:
        name = name.split(":")[0]

    # Map image names to component directories
    component_map = {
        "preprocessing": "components/preprocessing",
        "training-hmm": "components/training/hmm",
        "generation-hmm": "components/generation/hmm",
        "training-gaussian_process": "components/training/gaussian_process",
        "generation-gaussian_process": "components/generation/gaussian_process",
        "training-copulas": "components/training/copulas",
        "generation-copulas": "components/generation/copulas",
        "training-boltzman_machines": "components/training/boltzman_machines",
        "generation-boltzman_machines": "components/generation/boltzman_machines",
        "training-bayesian_networks": "components/training/bayesian_networks",
        "generation-bayesian_networks": "components/generation/bayesian_networks",
        "rl-gym-pybullet": "components/rl/gym-pybullet",
        "bouncing-ball": "components/unity/BouncingBall",
    }

    if name in component_map:
        return f"eval $(minikube docker-env) && docker build -t {name}:latest {component_map[name]}/"

    return None
