"""Notebook widget and client helpers for the local Ultralytics server.

This module exposes two student-facing building blocks:
1. UltralyticsManager: starts/stops Docker, monitors server health, and prepares models.
2. UltralyticsClient: sends images to the inference API and returns normalized detections.

The manager is intentionally notebook-friendly: it auto-starts in the background, surfaces
clear logs in a widget, and keeps operations idempotent so repeated button clicks are safe.
"""

import anywidget
import traitlets
import pathlib
import subprocess
import logging
import threading
import requests
import time
import psutil
import io
import re

from urllib.parse import quote

from PIL import Image


def defer(func):
    """
    Decorator to run a function in a background thread.

    This keeps notebook cells responsive while Docker operations run.
    """

    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper


class UltralyticsClient:
    """Small client wrapper for the local Ultralytics inference server.

    The public infer(image) API intentionally returns a compact schema that is easy for
    students to print and inspect in early computer-vision exercises.
    """

    def __init__(
        self,
        server_url: str,
        default_model: str,
        *,
        confidence: float = 0.4,
        iou: float = 0.45,
        image_size: int = 640,
        timeout: int = 10,
    ):
        self.server_url = server_url.rstrip("/")
        self.default_model = default_model
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.timeout = timeout

    def _to_jpeg_bytes(self, image) -> bytes:
        """Normalize common image input types to JPEG bytes accepted by the server."""
        if isinstance(image, (str, pathlib.Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        elif isinstance(image, bytes):
            return image
        else:
            try:
                import numpy as np

                if isinstance(image, np.ndarray):
                    if image.ndim == 2:
                        pil_image = Image.fromarray(image).convert("RGB")
                    else:
                        # OpenCV frames are usually BGR; flip to RGB.
                        pil_image = Image.fromarray(image[:, :, ::-1]).convert("RGB")
                else:
                    raise TypeError
            except Exception as exc:
                raise TypeError(
                    "Unsupported image type. Use a path, PIL image, numpy array, or JPEG bytes."
                ) from exc

        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def infer_raw(
        self,
        image,
        *,
        model: str | None = None,
        confidence: float | None = None,
        iou: float | None = None,
        image_size: int | None = None,
    ) -> dict:
        """Call the inference endpoint and return the raw JSON payload."""
        model_id = model or self.default_model
        payload = {
            "confidence": self.confidence if confidence is None else confidence,
            "iou": self.iou if iou is None else iou,
            "image_size": self.image_size if image_size is None else image_size,
        }
        image_bytes = self._to_jpeg_bytes(image)
        response = requests.post(
            f"{self.server_url}/infer/{quote(model_id, safe='')}",
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def infer(self, image, *, model: str | None = None) -> list[dict]:
        """Return a list of detected objects with confidences and bounding boxes."""
        raw = self.infer_raw(image, model=model)
        return [
            {
                "object": prediction["class"],
                "confidence": prediction["confidence"],
                "bbox": prediction["bbox"],
                "x": prediction["x"],
                "y": prediction["y"],
                "width": prediction["width"],
                "height": prediction["height"],
            }
            for prediction in raw.get("predictions", [])
        ]


class UltralyticsManager(anywidget.AnyWidget):
    """Widget-backed manager for Docker lifecycle and model orchestration.

    Notes for teaching environments:
    - Startup is automatic to reduce setup friction for first-time users.
    - All long-running Docker actions are executed on background threads.
    - Shared state updates are guarded by a lock to avoid races from concurrent actions.
    """

    # Frontend assets rendered by anywidget.
    _esm = pathlib.Path(__file__).parent / "ultralytics-manager.js"
    _css = pathlib.Path(__file__).parent / "ultralytics-manager.css"

    # Synced traitlets consumed by the JavaScript widget.
    status = traitlets.Unicode("...").tag(sync=True)
    logs = traitlets.Unicode("").tag(sync=True)
    memory_usage = traitlets.Unicode("0%").tag(sync=True)

    # Known states used by both Python and JS controls.
    _KNOWN_STATES = {
        "...",
        "starting",
        "running",
        "stopping",
        "stopped",
        "restarting",
        "error",
    }

    def __init__(
        self,
        verbose=False,
        server_url="http://localhost:18001",
        server_startup_timeout=120,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.server_url = server_url.rstrip("/")
        self.project_root = pathlib.Path(__file__).resolve().parent.parent
        self.server_startup_timeout = int(server_startup_timeout)

        level = logging.DEBUG if verbose else logging.INFO

        # Lock protects shared state touched by multiple worker threads.
        self._state_lock = threading.RLock()

        # Separate stop events avoid coupling Docker lifecycle to RAM reporting.
        self._docker_stop_event = threading.Event()
        self._ram_stop_event = threading.Event()

        # Keep references to background workers for clean joins/restarts.
        self._docker_monitor_thread: threading.Thread | None = None
        self._ram_monitor_thread: threading.Thread | None = None

        # Backward-compatible aliases for external notebooks that may inspect these attributes.
        self.docker_stop_event = self._docker_stop_event
        self.docker_monitor_task = None

        # Use a dedicated logger so notebook/kernel logs do not leak into widget logs.
        logger_name = f"ultralytics.manager.{id(self)}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()
        handler = logging.StreamHandler(self._LogStream(self))
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(handler)
        self.logger.propagate = False

        # Disable noisy HTTP/library logging
        for name in ("urllib3", "urllib3.connectionpool", "requests", "anywidget"):
            lib_logger = logging.getLogger(name)
            lib_logger.handlers.clear()
            lib_logger.propagate = False
            lib_logger.setLevel(logging.CRITICAL + 1)

        # Bootstrap visible widget values immediately so students never see a stale
        # "... / 0%" screen when threads are still warming up.
        self._set_status("starting")
        self._refresh_memory_usage_once()

        # Auto-start to preserve the current student notebook flow.
        self._start_docker_worker()
        self._start_ram_monitor_worker()

        self.on_msg(self._handle_frontend_message)

    # Custom logging handler to write logs into the traitlet shown in the widget.
    class _LogStream:
        def __init__(self, manager):
            self.manager = manager

        def write(self, message):
            self.manager._append_log(message)

        def flush(self):
            pass

    def _append_log(self, message: str):
        """Append non-empty log fragments with bounded history for widget rendering."""
        if not message or not message.strip():
            return
        with self._state_lock:
            self.logs += message
            if len(self.logs) > 5000:
                self.logs = self.logs[-5000:]

    def _set_status(self, new_status: str):
        """Thread-safe state update helper used by all lifecycle operations."""
        if new_status not in self._KNOWN_STATES:
            self.logger.warning(f"Ignoring unknown status '{new_status}'.")
            return
        with self._state_lock:
            if self.status != new_status:
                self.status = new_status

    def _get_status(self) -> str:
        """Return the current status under lock for consistent reads."""
        with self._state_lock:
            return self.status

    def _handle_frontend_message(self, _, content, buffers):
        """Dispatch actions sent from the JavaScript widget controls."""
        self._ensure_background_workers()
        action = content.get("action")
        if action == "restart_docker":
            self.restart_docker()
        elif action == "stop_docker":
            self.stop_docker()
        elif action == "start_docker":
            self.start_docker()

    def _run_compose(self, args: list[str]) -> subprocess.CompletedProcess:
        """Run compose from project root so notebooks work from any current directory."""
        commands = [
            ["docker", "compose", *args],
            ["docker-compose", *args],
        ]

        last_error = None
        for command in commands:
            try:
                return subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    cwd=str(self.project_root),
                )
            except FileNotFoundError as exc:
                last_error = exc

        raise RuntimeError(
            "Could not find Docker Compose. Install 'docker compose' or 'docker-compose'."
        ) from last_error

    def _extract_conflict_container_id(self, output: str) -> str | None:
        match = re.search(r'container "([0-9a-f]{12,64})"', output)
        return match.group(1) if match else None

    def _run_docker(self, args: list[str]) -> subprocess.CompletedProcess:
        """Run plain Docker CLI commands from project root."""
        return subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
        )

    def _list_running_inference_containers(self) -> list[str]:
        """Return running container names matching 'inference' from docker ps."""
        docker_ps = self._run_docker(
            ["ps", "--filter", "name=inference", "--format", "{{.Names}}"]
        )
        if docker_ps.returncode != 0:
            raise RuntimeError(
                "Failed to inspect containers with docker ps. "
                f"stderr:\n{docker_ps.stderr}"
            )
        return [name.strip() for name in docker_ps.stdout.splitlines() if name.strip()]

    def _is_inference_container_running(self) -> bool:
        """Return True when docker ps reports a running inference container."""
        return bool(self._list_running_inference_containers())

    def _join_docker_worker(self, timeout: float = 8.0):
        """Wait for current Docker worker to stop if it exists and is active."""
        with self._state_lock:
            worker = self._docker_monitor_thread
        if worker and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=timeout)

    def _start_docker_worker(self):
        """Start one Docker lifecycle worker if none is currently running."""
        with self._state_lock:
            if self._docker_monitor_thread and self._docker_monitor_thread.is_alive():
                return
            if self.status == "...":
                self.status = "starting"
            worker = threading.Thread(
                target=self._start_docker_in_background, daemon=True
            )
            self._docker_monitor_thread = worker
            # Preserve legacy attribute name for external notebook code.
            self.docker_monitor_task = worker
        worker.start()

    def _start_ram_monitor_worker(self):
        """Start one RAM reporting worker for the widget toolbar."""
        with self._state_lock:
            if self._ram_monitor_thread and self._ram_monitor_thread.is_alive():
                return
            worker = threading.Thread(target=self._update_ram_usage, daemon=True)
            self._ram_monitor_thread = worker
        worker.start()

    def _refresh_memory_usage_once(self):
        """Update memory usage once synchronously for immediate widget feedback."""
        try:
            with self._state_lock:
                self.memory_usage = f"{psutil.virtual_memory().percent}%"
        except Exception as exc:
            self.logger.error(f"Exception occurred while getting RAM usage: {exc}")

    def _ensure_background_workers(self):
        """Best-effort self-healing if background workers have unexpectedly stopped."""
        with self._state_lock:
            docker_alive = bool(
                self._docker_monitor_thread and self._docker_monitor_thread.is_alive()
            )
            ram_alive = bool(
                self._ram_monitor_thread and self._ram_monitor_thread.is_alive()
            )

        if not docker_alive and not self._docker_stop_event.is_set():
            self.logger.warning(
                "Docker lifecycle worker was not running; restarting it automatically."
            )
            self._start_docker_worker()
        if not ram_alive and not self._ram_stop_event.is_set():
            self.logger.warning(
                "RAM monitor worker was not running; restarting it automatically."
            )
            self._start_ram_monitor_worker()

    def _wait_for_server_ready(self) -> bool:
        """Poll /models until the API is ready or timeout expires.

        Container liveness is validated with docker ps on each cycle so we do not rely
        on internal events to infer whether the server is still running.
        """
        deadline = time.time() + self.server_startup_timeout
        while True:
            if time.time() > deadline:
                return False

            try:
                if not self._is_inference_container_running():
                    return False
            except Exception as exc:
                self.logger.error(
                    f"Unable to confirm container state via docker ps: {exc}"
                )
                return False

            try:
                response = requests.get(f"{self.server_url}/models", timeout=2)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                # During startup, connection failures are expected until the app is serving.
                pass
            time.sleep(1)

    def _start_docker_in_background(self):
        """
        Worker that ensures Docker is running and keeps monitoring container health.

        This runs on a dedicated thread managed by _start_docker_worker.
        """
        # Query docker ps first so startup decisions are based on actual container state.
        container_running = False
        try:
            container_running = self._is_inference_container_running()
        except Exception as exc:
            self.logger.warning(f"Could not query docker ps before startup: {exc}")

        # If the server API is already up, adopt running state immediately.
        try:
            response = requests.get(f"{self.server_url}/", timeout=2)
            if response.status_code == 200:
                self.logger.info("Docker container is already running.")
                self._set_status("running")
            else:
                self._set_status("starting")
        except requests.exceptions.RequestException:
            self._set_status("starting")

        if self._get_status() == "starting" and not container_running:
            self.logger.info(
                "Starting Docker container with 'docker compose up -d' "
                "(first build can take a few minutes)..."
            )
            try:
                result = self._run_compose(["up", "-d"])
                if result.returncode != 0:
                    combined_output = f"{result.stdout}\n{result.stderr}"
                    container_id = self._extract_conflict_container_id(combined_output)
                    if (
                        container_id
                        and "is already in use by container" in combined_output
                    ):
                        self.logger.warning(
                            "Detected stale container-name conflict. "
                            f"Removing container {container_id} and retrying startup once..."
                        )
                        rm_result = self._run_docker(["rm", "-f", container_id])
                        if rm_result.returncode != 0:
                            self.logger.error(
                                "Failed to remove conflicting container. "
                                f"stdout:\n{rm_result.stdout}\n"
                                f"stderr:\n{rm_result.stderr}"
                            )
                            self._set_status("error")
                            return
                        result = self._run_compose(["up", "-d"])

                if result.returncode != 0:
                    self.logger.error(
                        "Failed to start Docker container. "
                        f"cwd={self.project_root}\n"
                        f"stdout:\n{result.stdout}\n"
                        f"stderr:\n{result.stderr}"
                    )
                    self.logger.error(
                        "Tip: run 'docker compose logs inference' in the project root "
                        "for startup details."
                    )
                    self._set_status("error")
                    return
            except Exception as exc:
                self.logger.error(f"Exception while running Docker Compose: {exc}")
                self._set_status("error")
                return

            self.logger.info(
                f"Waiting for inference server readiness (timeout: {self.server_startup_timeout}s)..."
            )
            if not self._wait_for_server_ready():
                self.logger.error(
                    "Inference server did not become ready in time or container stopped. "
                    "Tip: run 'docker compose logs inference' from the project root."
                )
                self._set_status("error")
                return

            self.logger.info("Ultralytics inference server is ready.")
            self._set_status("running")

        if self._get_status() == "starting" and container_running:
            self.logger.info(
                "Container is already running per docker ps; waiting for API."
            )
            if not self._wait_for_server_ready():
                self.logger.error(
                    "Container is running but API did not become ready before timeout. "
                    "Tip: run 'docker compose logs inference' from the project root."
                )
                self._set_status("error")
                return
            self._set_status("running")

        # Monitor container state continuously using docker ps as source of truth.
        while True:
            try:
                active_names = self._list_running_inference_containers()
                if not active_names and self._get_status() not in {
                    "stopping",
                    "stopped",
                }:
                    if self._docker_stop_event.is_set():
                        self._set_status("stopped")
                        break
                    self.logger.error(
                        "Inference container appears to have stopped unexpectedly. "
                        "Use Start or Restart to recover."
                    )
                    self._set_status("error")
                    break
                if active_names and self._get_status() == "stopped":
                    self._set_status("running")
            except Exception as exc:
                self.logger.error(f"Exception while monitoring Docker container: {exc}")
                self._set_status("error")
                break

            if self._docker_stop_event.is_set() and not active_names:
                self._set_status("stopped")
                break

            time.sleep(5)

    @defer
    def restart_docker(self):
        """
        Restart Docker with a stop then start sequence.

        Runs in the background so notebooks remain responsive.
        """
        if self._get_status() == "restarting":
            self.logger.info("Restart already in progress. Ignoring duplicate request.")
            return

        self.logger.info("Restarting Docker container...")
        self._set_status("restarting")

        if not self._stop_docker_internal(mark_stopped=False):
            return

        self._docker_stop_event.clear()
        self._start_docker_worker()

    def _stop_docker_internal(self, mark_stopped: bool = True) -> bool:
        """Shared stop logic used by stop and restart flows."""
        self._docker_stop_event.set()

        try:
            result = self._run_compose(["down"])
            if result.returncode != 0:
                self.logger.error(
                    "Failed to stop Docker container cleanly. "
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
                self._set_status("error")
                self._join_docker_worker()
                return False
            self.logger.info("Docker container stopped successfully.")
        except Exception as e:
            self.logger.error(
                f"Exception occurred while stopping Docker container: {e}"
            )
            self._set_status("error")
            self._join_docker_worker()
            return False

        self._join_docker_worker()
        if mark_stopped:
            try:
                if self._is_inference_container_running():
                    self.logger.error(
                        "docker compose down completed but docker ps still shows an "
                        "inference container running."
                    )
                    self._set_status("error")
                    return False
            except Exception as exc:
                self.logger.error(f"Could not verify stop state with docker ps: {exc}")
                self._set_status("error")
                return False
            self._set_status("stopped")
        return True

    @defer
    def stop_docker(self):
        """
        Stop Docker if running.

        This operation is idempotent and safe to call repeatedly.
        """
        if self._get_status() in {"stopped", "stopping"}:
            self.logger.info("Docker is already stopping or stopped.")
            return

        self.logger.info("Stopping Docker container...")
        self._set_status("stopping")
        self._stop_docker_internal(mark_stopped=True)

    @defer
    def start_docker(self):
        """
        Start Docker if not already running.

        This clears previous stop requests and launches one lifecycle worker.
        """
        if self._get_status() in {"running", "starting", "restarting"}:
            self.logger.info("Docker is already starting or running.")
            return

        self.logger.info("Starting Docker container...")
        self._docker_stop_event.clear()
        self._start_docker_worker()

    def _update_ram_usage(self):
        """
        Periodically checks system RAM usage and updates the memory_usage trait. This is done in a
        background thread to avoid blocking the main thread.
        """
        while not self._ram_stop_event.is_set():
            try:
                with self._state_lock:
                    self.memory_usage = f"{psutil.virtual_memory().percent}%"
            except Exception as e:
                self.logger.error(f"Exception occurred while getting RAM usage: {e}")

            time.sleep(1)

    def _wait_until_running(self):
        """Block until running or raise a clear error when startup fails/stops."""
        self._ensure_background_workers()

        if self._get_status() == "running":
            return

        print("<ultralytics> Waiting for inference server to be running...")
        while self._get_status() != "running":
            self._ensure_background_workers()
            current_status = self._get_status()
            if current_status == "error":
                raise RuntimeError(
                    "Inference server failed to start. Check widget logs and run "
                    "'docker compose logs inference' from the project root for details."
                )
            if current_status == "stopped":
                raise RuntimeError(
                    "Docker container was stopped before the server could start."
                )

            try:
                if not self._is_inference_container_running() and current_status in {
                    "stopping",
                    "stopped",
                }:
                    raise RuntimeError(
                        "Docker container is not running according to docker ps. "
                        "Use Start to launch it again."
                    )
            except RuntimeError:
                raise
            except Exception as exc:
                self.logger.warning(
                    f"Could not verify container state via docker ps while waiting: {exc}"
                )
            time.sleep(1)

    def _list_models_state(self) -> dict:
        """Return available and loaded model state from the server."""
        try:
            response = requests.get(f"{self.server_url}/models", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                "Could not read model state from the inference server. "
                "Ensure Docker is running and the server is healthy."
            ) from exc

    def _load_model(self, model: str):
        """Request model load from the server."""
        response = requests.post(
            f"{self.server_url}/models/{quote(model, safe='')}/load", timeout=30
        )
        response.raise_for_status()

    def _unload_model(self, model: str):
        """Request model unload; keep operation idempotent for easier student workflows."""
        response = requests.delete(
            f"{self.server_url}/models/{quote(model, safe='')}/unload", timeout=30
        )
        # Ignore unloads for models that are already not loaded.
        if response.status_code not in (200, 404):
            response.raise_for_status()

    def _ensure_models_exist(self, models: list[str]):
        """Download missing .pt models from Ultralytics to ./models."""
        try:
            from ultralytics import YOLO
            import shutil
            import urllib.request
        except ImportError:
            self.logger.warning(
                "Ultralytics library not available for auto-download. "
                "Models must be manually placed in ./models/"
            )
            return

        models_dir = self.project_root / "models"
        models_dir.mkdir(exist_ok=True)

        for model_name in models:
            model_path = models_dir / model_name
            if model_path.exists():
                self.logger.info(
                    f"Model '{model_name}' already present at {model_path}"
                )
                continue

            self.logger.info(f"Model '{model_name}' not found locally. Downloading...")
            try:
                # First, try the standard Ultralytics hub download approach
                YOLO(model_name)

                # YOLO objects have a .model_name or we can get the path from the model
                # Try common cache locations
                possible_paths = [
                    pathlib.Path.home() / ".cache" / "ultralytics" / "hub" / model_name,
                    pathlib.Path.home()
                    / ".cache"
                    / "ultralytics"
                    / "models"
                    / model_name,
                    pathlib.Path.home() / ".cache" / "ultralytics" / model_name,
                    pathlib.Path.home() / ".ultralytics" / "models" / model_name,
                ]

                downloaded = False
                for cache_path in possible_paths:
                    if cache_path.exists():
                        shutil.copy2(cache_path, model_path)
                        self.logger.info(
                            f"Copied {model_name} from cache to {model_path}"
                        )
                        downloaded = True
                        break

                if not downloaded:
                    # Fall back to direct release download when local cache lookup fails.
                    self.logger.info(
                        f"Attempting direct download from Ultralytics hub..."
                    )
                    url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}"
                    urllib.request.urlretrieve(url, model_path)
                    self.logger.info(
                        f"Downloaded {model_name} directly to {model_path}"
                    )

            except Exception as e:
                self.logger.error(
                    f"Failed to download {model_name}: {e}. "
                    f"Please manually download from https://github.com/ultralytics/assets "
                    f"and place in {models_dir}/"
                )

    def create_client(
        self,
        models: list[str] | None = None,
        *,
        default_model: str | None = None,
        confidence: float = 0.4,
        iou: float = 0.45,
        image_size: int = 640,
        timeout: int = 10,
    ):
        """
        Wait for server readiness, download missing models, load selected models,
        unload non-selected loaded models, then return a client with a simple infer(image) API.
        """

        self._wait_until_running()

        # Download missing models before attempting to load them
        if models:
            self._ensure_models_exist(models)

        model_state = self._list_models_state()
        available_models = model_state.get("models", [])
        loaded_models = model_state.get("loaded", [])

        selected_models = list(dict.fromkeys(models or []))
        if not selected_models:
            if not available_models:
                raise RuntimeError(
                    "No local .pt models found in ./models. Add a model before creating a client."
                )
            selected_models = [available_models[0]]
            self.logger.info(
                f"No models were specified. Defaulting to first available model '{selected_models[0]}'."
            )

        for model in selected_models:
            if model in loaded_models:
                continue
            try:
                self.logger.info(f"Loading model '{model}'...")
                self._load_model(model)
                self.logger.info(f"Model '{model}' loaded.")
            except Exception as e:
                # If the model fails to load and isn't in available_models, provide a helpful error
                if model not in available_models:
                    raise RuntimeError(
                        f"Model '{model}' not found or failed to load. "
                        f"Available models: {', '.join(available_models) or 'none'}. "
                        f"Ensure the model file exists in ./models/. Error: {e}"
                    ) from e
                else:
                    # If it is in available_models but still failed to load, propagate the error
                    raise

        for model in loaded_models:
            if model in selected_models:
                continue
            self.logger.info(f"Unloading model '{model}' since it was not selected...")
            self._unload_model(model)

        chosen_default = default_model or selected_models[0]
        if chosen_default not in selected_models:
            raise RuntimeError(
                f"default_model '{chosen_default}' must be one of selected models: {selected_models}"
            )

        self.logger.info(
            f"Ready! Selected models: {', '.join(selected_models)}. Default model: {chosen_default}"
        )

        return UltralyticsClient(
            server_url=self.server_url,
            default_model=chosen_default,
            confidence=confidence,
            iou=iou,
            image_size=image_size,
            timeout=timeout,
        )

    def close(self):
        """Best-effort cleanup when widget is disposed by the notebook."""
        self._docker_stop_event.set()
        self._ram_stop_event.set()
        self._join_docker_worker(timeout=2.0)

        with self._state_lock:
            ram_worker = self._ram_monitor_thread
        if (
            ram_worker
            and ram_worker.is_alive()
            and ram_worker is not threading.current_thread()
        ):
            ram_worker.join(timeout=2.0)

        super().close()
