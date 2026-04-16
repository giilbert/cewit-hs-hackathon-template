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
    }
    model.on("change:logs", updateLogs);
    updateLogs();

    const memoryUsage = document.createElement("div");
    const updateMemoryUsage = () => memoryUsage.innerText = `RAM Usage: ${model.get("memory_usage")}`;
    model.on("change:memory_usage", updateMemoryUsage);
    updateMemoryUsage();
    memoryUsage.classList.add("memory-usage");
    toolbar.appendChild(memoryUsage);

    const restartButton = document.createElement("button");
    restartButton.innerText = "Restart";
    restartButton.addEventListener("click", () => {
        restartButton.disabled = true;
        model.send({ action: "restart_docker" });
    });
    model.on("change:status", () => {
        restartButton.disabled = model.get("status") === "restarting";
    });
    restartButton.classList.add("restart-button");
    toolbar.appendChild(restartButton);

    el.classList.add("roboflow-manager");
}

export default { render };