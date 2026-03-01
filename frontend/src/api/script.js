// Detect API base URL: if running in browser on host, use host IP; if running in container, use service name
const API_BASE = (() => {
    // If running in browser and hostname is 127.0.0.1 or localhost, it's on the host machine
    // Use 127.0.0.1:8000 for local development
    // If hostname is something else (e.g. from nginx in docker), try to infer the API location
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    const apiPort = 8000;
    // For now, always use the host's gateway port
    return `${protocol}//${hostname}:${apiPort}/api/v1`;
})();


const fileInput = document.getElementById("fileInput");
const videoPlayer = document.getElementById("videoPlayer");
const removeBtn = document.getElementById("removeBtn");
const uploadBtn = document.getElementById("uploadBtn");

const identifyForm = document.getElementById("identify-form");
const nicknameInput = document.getElementById("nickname-input");
const userStatus = document.getElementById("user-status");

const searchForm = document.getElementById("search-form");
const promptInput = document.getElementById("prompt-input");
const searchStatus = document.getElementById("search-status");
const resultsList = document.getElementById("results-list");
const videoStatus = document.getElementById("video-status");

let currentVideoUrl = null;
let selectedFile = null;
let userId = null;
let videoId = null;

// Supported HTML5 video MIME types
const SUPPORTED_VIDEO_TYPES = [
    "video/mp4", "video/webm", "video/ogg",
    "video/quicktime",
];

// Human-readable status names
const STATUS_LABELS = {
    pending: "ожидание",
    indexing: "индексация...",
    ready: "готово",
    completed: "завершено",
    not_found: "ничего не найдено",
    failed: "ошибка",
};

function statusLabel(raw) {
    return STATUS_LABELS[raw] || raw;
}

function setStatus(el, text, cssClass) {
    el.textContent = text;
    el.className = "";
    if (cssClass) el.classList.add(cssClass);
}

function statusClass(raw) {
    if (raw === "ready" || raw === "completed") return "status-ready";
    if (raw === "pending" || raw === "indexing") return "status-pending";
    if (raw === "failed" || raw === "error") return "status-failed";
    return "";
}

function formatTimestamp(seconds) {
    if (seconds == null || isNaN(seconds)) return "??:??";
    const totalSec = Math.floor(seconds);
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function formatTimeRange(startSec, endSec) {
    return `${formatTimestamp(startSec)} – ${formatTimestamp(endSec)}`;
}

function seekToTimestamp(seconds) {
    if (videoPlayer && seconds != null && !isNaN(seconds)) {
        videoPlayer.currentTime = seconds;
        videoPlayer.hidden = false;
        videoPlayer.play().catch(() => {});
    }
}

/**
 * Build a result <li> with a play button and interval info.
 */
function buildResultItem(item) {
    const li = document.createElement("li");
    li.className = "result-item";

    const startTime = item.start_time;
    const endTime = item.end_time;
    const seekTime = item.start_time ?? item.timestamp ?? item.time ?? item.t;
    const score = item.score ?? item.similarity_score;
    const caption = item.caption;

    // Play button
    const btn = document.createElement("button");
    btn.className = "result-seek-btn";
    btn.title = `Перейти к ${formatTimestamp(seekTime)}`;
    btn.textContent = "\u25B6";
    btn.addEventListener("click", (e) => {
        e.stopPropagation();
        seekToTimestamp(seekTime);
    });
    li.appendChild(btn);

    // Text block
    const textDiv = document.createElement("div");
    textDiv.className = "result-text";

    const timeLine = document.createElement("div");
    timeLine.className = "result-time";
    if (startTime != null && endTime != null) {
        timeLine.textContent = formatTimeRange(startTime, endTime);
    } else {
        const ts = item.timestamp ?? item.time ?? item.t;
        timeLine.textContent = ts != null ? formatTimestamp(ts) : "\u2014";
    }
    textDiv.appendChild(timeLine);

    const parts = [];
    if (score != null) parts.push(`score: ${score}`);
    if (caption) parts.push(caption);
    if (parts.length) {
        const metaLine = document.createElement("div");
        metaLine.className = "result-meta";
        metaLine.textContent = parts.join(" \u2014 ");
        textDiv.appendChild(metaLine);
    }

    li.appendChild(textDiv);

    li.style.cursor = "pointer";
    li.addEventListener("click", () => seekToTimestamp(seekTime));

    return li;
}

function setResults(items) {
    resultsList.innerHTML = "";
    if (!items || items.length === 0) {
        const li = document.createElement("li");
        li.textContent = "Ничего не найдено";
        li.style.color = "var(--muted)";
        resultsList.appendChild(li);
        return;
    }
    items.forEach((item) => {
        if (item && typeof item === "object") {
            resultsList.appendChild(buildResultItem(item));
        } else {
            const li = document.createElement("li");
            li.textContent = typeof item === "number" ? formatTimestamp(item) : String(item);
            resultsList.appendChild(li);
        }
    });
}

// --- Video error handling ---
videoPlayer.addEventListener("error", () => {
    const err = videoPlayer.error;
    if (err) {
        const messages = {
            1: "Воспроизведение прервано",
            2: "Сетевая ошибка загрузки видео",
            3: "Ошибка декодирования видео \u2014 формат не поддерживается браузером",
            4: "Формат видео не поддерживается",
        };
        const msg = messages[err.code] || `Ошибка видеоплеера (код ${err.code})`;
        setStatus(videoStatus, msg, "status-failed");
        console.warn("[VideoPlayer]", msg, err.message);
    }
});

async function requestJson(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `HTTP ${response.status}`);
    }
    return response.json();
}

// ======================== IDENTIFY ========================

identifyForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const nickname = nicknameInput.value.trim();
    if (!nickname) return;
    setStatus(userStatus, "идёт запрос...", "status-pending");
    try {
        const data = await requestJson(`${API_BASE}/identify`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ nickname })
        });
        userId = data.user_id;
        setStatus(userStatus, `${data.nickname} (id: ${userId})`, "status-ready");
    } catch (err) {
        setStatus(userStatus, "ошибка", "status-failed");
        alert(`Ошибка идентификации: ${err.message}`);
    }
});

// ======================== VIDEO ========================

fileInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) {
        selectedFile = file;
        const mimeOk = file.type && SUPPORTED_VIDEO_TYPES.includes(file.type);
        if (!mimeOk && file.type) {
            setStatus(videoStatus, `\u26A0 Формат ${file.type} может не воспроизводиться`, "status-pending");
        } else {
            setStatus(videoStatus, "готово к загрузке", "");
        }
        if (currentVideoUrl) {
            URL.revokeObjectURL(currentVideoUrl);
        }
        currentVideoUrl = URL.createObjectURL(file);
        videoPlayer.src = currentVideoUrl;
        videoPlayer.hidden = false;
        videoPlayer.load();
        removeBtn.hidden = false;
    }
});

removeBtn.addEventListener("click", () => {
    videoPlayer.pause();
    videoPlayer.src = "";
    videoPlayer.hidden = true;
    removeBtn.hidden = true;
    if (currentVideoUrl) {
        URL.revokeObjectURL(currentVideoUrl);
        currentVideoUrl = null;
    }
    selectedFile = null;
    videoId = null;
    setStatus(videoStatus, "не загружено", "");
});

uploadBtn.addEventListener("click", async () => {
    if (!userId) {
        alert("Сначала введите никнейм.");
        return;
    }
    if (!selectedFile) {
        alert("Выберите видео.");
        return;
    }
    setStatus(videoStatus, "загрузка на сервер...", "status-pending");
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("user_id", String(userId));
    try {
        const data = await requestJson(`${API_BASE}/videos`, {
            method: "POST",
            body: formData
        });
        videoId = data.video_id;
        setStatus(videoStatus, `загружено (id: ${videoId}) \u2014 ${statusLabel(data.status)}`, statusClass(data.status));
        // Start polling for indexing completion
        pollVideoStatus(videoId);
    } catch (err) {
        setStatus(videoStatus, "ошибка загрузки", "status-failed");
        alert(`Ошибка загрузки: ${err.message}`);
    }
});

/**
 * Poll GET /videos/{id} until processing_status is terminal.
 */
async function pollVideoStatus(id) {
    const terminalStatuses = ["ready", "completed", "failed"];
    const maxTries = 120;

    for (let i = 0; i < maxTries; i++) {
        await new Promise((r) => setTimeout(r, 2000));
        try {
            const data = await requestJson(`${API_BASE}/videos/${id}`, { method: "GET" });
            const st = data.status;
            setStatus(videoStatus, `видео (id: ${id}) \u2014 ${statusLabel(st)}`, statusClass(st));
            if (terminalStatuses.includes(st)) {
                return;
            }
        } catch (err) {
            console.warn("[PollVideo]", err.message);
        }
    }
    setStatus(videoStatus, `видео (id: ${id}) \u2014 таймаут ожидания`, "status-failed");
}

// ======================== SEARCH ========================

searchForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!userId || !videoId) {
        alert("Нужны пользователь и загруженное видео.");
        return;
    }
    const queryText = promptInput.value.trim();
    if (!queryText) return;
    setStatus(searchStatus, "отправка...", "status-pending");
    setResults([]);
    try {
        const data = await requestJson(`${API_BASE}/searches`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                user_id: userId,
                video_id: videoId,
                query_text: queryText
            })
        });
        const queryId = data.query_id;
        setStatus(searchStatus, `в обработке... (id: ${queryId})`, "status-pending");
        await pollSearch(queryId);
    } catch (err) {
        setStatus(searchStatus, "ошибка", "status-failed");
        alert(`Ошибка поиска: ${err.message}`);
    }
});

async function pollSearch(queryId) {
    const maxTries = 120;
    const terminalStatuses = ["ready", "completed", "not_found", "failed"];

    for (let i = 0; i < maxTries; i++) {
        try {
            const statusData = await requestJson(`${API_BASE}/searches/${queryId}`, {
                method: "GET"
            });
            const st = statusData.status;
            setStatus(searchStatus, statusLabel(st), statusClass(st));

            if (terminalStatuses.includes(st)) {
                if (st === "ready" || st === "completed") {
                    try {
                        const resultsData = await requestJson(`${API_BASE}/searches/${queryId}/results`, {
                            method: "GET"
                        });
                        const result = Array.isArray(resultsData.result)
                            ? resultsData.result
                            : Array.isArray(resultsData.results)
                                ? resultsData.results
                                : [];
                        setResults(result);
                    } catch (err) {
                        setResults([]);
                        setStatus(searchStatus, "ошибка получения результатов", "status-failed");
                    }
                } else if (st === "not_found") {
                    setResults([]);
                } else if (st === "failed") {
                    setResults([]);
                    setStatus(searchStatus, "ошибка обработки", "status-failed");
                }
                return;
            }
        } catch (err) {
            console.warn("[PollSearch]", err.message);
        }

        await new Promise((resolve) => setTimeout(resolve, 2000));
    }
    setStatus(searchStatus, "таймаут ожидания", "status-failed");
}