"""
A file of useful stuff for interacting with Roboflow in the notebook.

DO NOT DELETE!!!
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

from inference_sdk import InferenceHTTPClient


def defer(func):
    """
    Decorator to run a function in the background using subprocess.
    """

    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper


class RoboflowManager(anywidget.AnyWidget):
    # The frontend JavaScript logic
    _esm = pathlib.Path(__file__).parent / "roboflow-manager.js"
    _css = pathlib.Path(__file__).parent / "roboflow-manager.css"

    status = traitlets.Unicode("...").tag(sync=True)
    logs = traitlets.Unicode("").tag(sync=True)
    memory_usage = traitlets.Unicode("0%").tag(sync=True)

    def __init__(self, verbose=False, roboflow_api_key=None, **kwargs):
        super().__init__(**kwargs)

        self.roboflow_api_key = roboflow_api_key

        level = logging.DEBUG if verbose else logging.INFO

        # Capture all logs through the root logger, but do not print to stdout/stderr
        self.logger = logging.getLogger()
        self.logger.setLevel(level)
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
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

        self.docker_stop_event = threading.Event()
        self.docker_monitor_task = self._start_docker_in_background()
        self._update_ram_usage()

        self.on_msg(self._handle_frontend_message)

    # Custom logging handler to write logs to the logs trait
    class _LogStream:
        def __init__(self, manager):
            self.manager = manager

        def write(self, message):
            if message.strip():  # Avoid adding empty lines
                self.manager.logs += message

                # Keep only the last 5,000 characters
                if len(self.manager.logs) > 5000:
                    self.manager.logs = self.manager.logs[-5000:]

        def flush(self):
            pass

    def _handle_frontend_message(self, _, content, buffers):
        if content.get("action") == "restart_docker":
            self.restart_docker()

    @defer
    def _start_docker_in_background(self):
        """
        Runs `docker compose up -d` and updates the status to "starting" and then "running"
        """

        # Check if the Docker container is already running by attempting to connect to the server
        try:
            response = requests.get("http://localhost:9001/health")
            if response.status_code == 200:
                self.logger.info("Docker container is already running.")
                self._set_status("running")
                return
        except requests.exceptions.ConnectionError:
            self.logger.info("Docker container is not running. Attempting to start it.")

        self.logger.info("Starting Docker container with `docker compose up -d`")
        self._set_status("starting")

        try:
            result = subprocess.run(
                ["docker", "compose", "up", "-d"], capture_output=True, text=True
            )
            if result.returncode == 0:
                self.logger.info("Docker container started successfully. ")
            else:
                self.logger.error(f"Failed to start Docker container:\n{result.stderr}")
                self._set_status("error")
                return

            # Wait until the Roboflow inference server is ready to accept requests
            self.logger.info("Waiting for the Roboflow inference server to be ready...")
            while True:
                if self.docker_stop_event.is_set():
                    break

                try:
                    response = requests.get("http://localhost:9001/model/registry")
                    if response.status_code == 200:
                        self.logger.info("Roboflow inference server is ready!")
                        self._set_status("running")
                        break
                except requests.exceptions.ConnectionError:
                    pass

                time.sleep(1)
        except Exception as e:
            self.logger.error(
                f"Exception occurred while starting Docker container: {e}"
            )
            self._set_status("error")

        # Docker has started and the server is ready, now we need to monitor the Docker container
        # and update the status if it stops for any reason
        while True:
            if self.docker_stop_event.is_set():
                break

            try:
                docker_ps = subprocess.run(
                    ["docker", "ps", "--filter", "name=inference-server"],
                    capture_output=True,
                    text=True,
                )
                if docker_ps.returncode != 0:
                    self.logger.error("Failed to check Docker container status.")
                    self._set_status("error")
            except Exception as e:
                self.logger.error(
                    f"Exception occurred while checking Docker container status: {e}"
                )
                self._set_status("error")
                break

            time.sleep(5)  # Check every 5 seconds

    def restart_docker(self):
        """
        Public method to restart the Docker container. This can be called from the frontend when the user clicks a "Restart" button.
        """
        self.logger.info("Restarting Docker container...")
        self._set_status("restarting")

        try:
            subprocess.run(
                ["docker", "compose", "down"], capture_output=True, text=True
            )
            self.logger.info("Docker container stopped successfully.")
        except Exception as e:
            self.logger.error(
                f"Exception occurred while stopping Docker container: {e}"
            )
            self._set_status("error")
            return

        self.docker_stop_event.set()  # Signal the monitoring thread to stop
        self.docker_monitor_task.join()  # Wait for the previous monitoring thread to finish
        self.docker_stop_event.clear()  # Clear the stop event for the next monitoring thread

        # Start the Docker container again in the background
        self.docker_monitor_task = self._start_docker_in_background()

    def _set_status(self, new_status):
        self.status = new_status

    @defer
    def _update_ram_usage(self):
        """
        Periodically checks system RAM usage and updates the memory_usage trait. This is done in a
        background thread to avoid blocking the main thread.
        """
        while True:
            try:
                self.memory_usage = f"{psutil.virtual_memory().percent}%"
            except Exception as e:
                self.logger.error(f"Exception occurred while getting RAM usage: {e}")

            time.sleep(1)  # Update every second

    def create_client(self, models: list[str] | None = None):
        """
        Waits for the inference server to be ready to accept requests for specified models, then
        creates and returns an InferenceHTTPClient instance connected to the server.
        """

        client = InferenceHTTPClient(
            api_url="http://localhost:9001", api_key=self.roboflow_api_key
        )

        # Wait until the server is ready to accept requests
        if self.status != "running":
            print("<roboflow> Waiting for inference server to be running...")
            self.logger.info(
                f"Waiting for the server to be running before selecting models '{models}'..."
            )
            while self.status != "running":
                if self.status == "error":
                    raise RuntimeError(
                        "Inference server failed to start. Check logs for details."
                    )
                if self.docker_stop_event.is_set():
                    raise RuntimeError(
                        "Docker container was stopped before the server could start."
                    )
                time.sleep(1)

        registered_models = client.list_loaded_models()

        # Load requested models if they are not already loaded, and wait for them to be ready
        for model in models or []:
            if any(m.model_id == model for m in registered_models.models):
                continue

            self.logger.info(f"Waiting for model '{model}' to be available...")
            try:
                client.load_model(model)
            except Exception as e:
                self.logger.error(f"Error loading model '{model}': {e}")
                raise

        # Unload any models that were not requested to be loaded, to free up RAM
        for registered_model in registered_models.models:
            id = registered_model.model_id
            if models and id not in models:
                self.logger.info(
                    f"Unloading model '{id}' since it was not requested to be loaded..."
                )
                try:
                    client.unload_model(id)
                    self.logger.info(f"Model '{id}' unloaded successfully!")
                except Exception as e:
                    self.logger.error(f"Error unloading model '{id}': {e}")
                    raise

        self.logger.info(
            f"Done loading and unloading models! List: {', '.join([m.model_id for m in client.list_loaded_models().models])}"
        )

        return client
