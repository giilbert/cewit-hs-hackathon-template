function render({ model, el }) {
    const toolbar = document.createElement("div");
    toolbar.classList.add("toolbar");
    el.appendChild(toolbar);

    const status = document.createElement("div");
    const updateStatus = () => status.innerText = `${model.get("status").toUpperCase()}`;
    model.on("change:status", updateStatus);
    updateStatus();
    status.classList.add("status");
    toolbar.appendChild(status);

    const logs = document.createElement("pre");
    logs.classList.add("logs");
    el.appendChild(logs);
    const updateLogs = () => {
        logs.innerText = model.get("logs");
        if (logs.innerText !== "") logs.style.height = "200px";
        if (logs.scrollHeight - logs.scrollTop - logs.clientHeight < 50)
            logs.scrollTop = logs.scrollHeight;
    };
    model.on("change:logs", updateLogs);
    updateLogs();

    const memoryUsage = document.createElement("div");
    const updateMemoryUsage = () => memoryUsage.innerText = `RAM Usage: ${model.get("memory_usage")}`;
    model.on("change:memory_usage", updateMemoryUsage);
    updateMemoryUsage();
    memoryUsage.classList.add("memory-usage");
    toolbar.appendChild(memoryUsage);

    const stopStartButton = document.createElement("button");
    const updateStopStartButton = () => {
        const status = model.get("status");
        if (status === "stopped") {
            stopStartButton.innerText = "Start";
            stopStartButton.disabled = false;
        } else if (status === "running" || status === "..." || status === "error") {
            stopStartButton.innerText = "Stop";
            stopStartButton.disabled = false;
        } else {
            // stopping, starting, restarting
            stopStartButton.innerText = status.charAt(0).toUpperCase() + status.slice(1) + "...";
            stopStartButton.disabled = true;
        }
    };
    stopStartButton.addEventListener("click", () => {
        const status = model.get("status");
        stopStartButton.disabled = true;
        if (status === "stopped") {
            model.send({ action: "start_docker" });
        } else {
            model.send({ action: "stop_docker" });
        }
    });
    model.on("change:status", updateStopStartButton);
    updateStopStartButton();
    stopStartButton.classList.add("stop-start-button");
    toolbar.appendChild(stopStartButton);

    const restartButton = document.createElement("button");
    restartButton.innerText = "Restart";
    restartButton.addEventListener("click", () => {
        restartButton.disabled = true;
        model.send({ action: "restart_docker" });
    });
    model.on("change:status", () => {
        const status = model.get("status");
        restartButton.disabled = status === "restarting" || status === "stopped";
    });
    restartButton.classList.add("restart-button");
    toolbar.appendChild(restartButton);

    el.classList.add("ultralytics-manager");
}

export default { render };