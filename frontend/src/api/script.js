const API_BASE = (() => {
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    const apiPort = 8000;
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
const dropArea = document.getElementById("dropArea");
const videoStage = document.getElementById("video-stage");
const bboxOverlay = document.getElementById("bboxOverlay");
const bboxToggle = document.getElementById("bboxToggle");
const searchSubmitBtn = searchForm.querySelector('button[type="submit"]');

let currentVideoUrl = null;
let selectedFile = null;
let userId = null;
let videoId = null;
let videoProcessingStatus = null;
let isUploadingVideo = false;
let isSearching = false;
let videoStateVersion = 0;
let searchStateVersion = 0;
let activeOverlayBBox = null;
let activeOverlayLabel = "";
let bboxOverlayEnabled = true;

if (videoStage) {
    videoStage.hidden = true;
}
if (videoPlayer) {
    videoPlayer.hidden = true;
}

const SUPPORTED_VIDEO_TYPES = [
    "video/mp4", "video/webm", "video/ogg",
    "video/quicktime",
];

const STATUS_LABELS = {
    pending: "Ожидание",
    indexing: "Индексация...",
    ready: "Готово",
    completed: "Завершено",
    not_found: "Ничего не найдено",
    failed: "Ошибка",
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

function isVideoReadyForSearch() {
    return videoProcessingStatus === "ready" || videoProcessingStatus === "completed";
}

function syncUploadControls() {
    const isLocked = isUploadingVideo || videoId !== null;
    const hasVideo = Boolean(selectedFile || videoId || currentVideoUrl);

    fileInput.disabled = isLocked;
    uploadBtn.disabled = !selectedFile || isLocked;
    removeBtn.hidden = !hasVideo;
    removeBtn.disabled = isUploadingVideo || !hasVideo;

    if (dropArea) {
        dropArea.classList.toggle("is-disabled", isLocked);
        dropArea.setAttribute("aria-disabled", String(isLocked));
    }
}

function syncSearchControls() {
    searchSubmitBtn.disabled = isSearching || !userId || !videoId || !isVideoReadyForSearch();
}

function revokeCurrentVideoUrl() {
    if (currentVideoUrl) {
        URL.revokeObjectURL(currentVideoUrl);
        currentVideoUrl = null;
    }
}

function resetVideoPreview() {
    videoPlayer.pause();
    videoPlayer.removeAttribute("src");
    videoPlayer.load();
    videoPlayer.hidden = true;
    if (videoStage) videoStage.hidden = true;
    resetBboxOverlayState();
    revokeCurrentVideoUrl();
}

function resetSearchUi() {
    searchStateVersion += 1;
    isSearching = false;
    setStatus(searchStatus, "ожидание", "");
    resetBboxOverlayState();
    resultsList.innerHTML = "";
    syncSearchControls();
}

function formatTimestamp(seconds) {
    if (seconds == null || Number.isNaN(Number(seconds))) return "??:??";
    const totalSec = Math.floor(Number(seconds));
    const h = Math.floor(totalSec / 3600);
    const m = Math.floor((totalSec % 3600) / 60);
    const s = totalSec % 60;
    if (h > 0) {
        return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    }
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function formatTimeRange(startSec, endSec) {
    return `${formatTimestamp(startSec)} – ${formatTimestamp(endSec)}`;
}

function toNumberOrNull(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
}

function normalizeBbox(value) {
    if (Array.isArray(value)) {
        return value.map((v) => Number(v)).filter((v) => Number.isFinite(v));
    }
    if (typeof value === "string") {
        try {
            const parsed = JSON.parse(value);
            if (Array.isArray(parsed)) {
                return parsed.map((v) => Number(v)).filter((v) => Number.isFinite(v));
            }
        } catch {
            return [];
        }
    }
    return [];
}

function normalizeResultItem(item) {
    const start = toNumberOrNull(item.start ?? item.start_time ?? item.segment_start);
    const end = toNumberOrNull(item.end ?? item.end_time ?? item.segment_end);
    const bestTs = toNumberOrNull(item.best_ts ?? item.bestTs ?? item.timestamp ?? item.time ?? item.t ?? start);
    const score = toNumberOrNull(item.score ?? item.similarity_score);
    const hitType = item.type ?? item.hit_type ?? item?.yolo_metadata?.type ?? item?.metadata?.type;
    const bbox = normalizeBbox(item.bbox ?? item?.yolo_metadata?.bbox ?? item?.metadata?.bbox);

    return {
        start,
        end,
        bestTs,
        score,
        hitType,
        bbox,
    };
}

function clearBboxOverlay() {
    if (!bboxOverlay) return;
    bboxOverlay.innerHTML = "";
    bboxOverlay.hidden = true;
}

function resetBboxOverlayState() {
    activeOverlayBBox = null;
    activeOverlayLabel = "";
    clearBboxOverlay();
}

function refreshBboxOverlay() {
    if (!bboxOverlayEnabled) {
        if (bboxOverlay) {
            bboxOverlay.innerHTML = "";
            bboxOverlay.hidden = true;
        }
        return;
    }

    renderBboxOverlay();
}

function renderBboxOverlay() {
    if (!bboxOverlayEnabled || !bboxOverlay || !videoPlayer || !activeOverlayBBox || activeOverlayBBox.length !== 4) {
        return;
    }

    const [x1, y1, x2, y2] = activeOverlayBBox;
    const naturalWidth = Number(videoPlayer.videoWidth);
    const naturalHeight = Number(videoPlayer.videoHeight);
    const renderedWidth = videoPlayer.clientWidth;
    const renderedHeight = videoPlayer.clientHeight;

    if (!naturalWidth || !naturalHeight || !renderedWidth || !renderedHeight) {
        return;
    }

    const stageRect = videoStage ? videoStage.getBoundingClientRect() : null;
    const videoRect = videoPlayer.getBoundingClientRect();
    const displayedRatio = naturalWidth / naturalHeight;
    const boxRatio = renderedWidth / renderedHeight;

    let contentWidth = renderedWidth;
    let contentHeight = renderedHeight;
    let contentOffsetX = 0;
    let contentOffsetY = 0;

    if (Math.abs(boxRatio - displayedRatio) > 0.001) {
        if (boxRatio > displayedRatio) {
            contentHeight = renderedHeight;
            contentWidth = renderedHeight * displayedRatio;
            contentOffsetX = (renderedWidth - contentWidth) / 2;
        } else {
            contentWidth = renderedWidth;
            contentHeight = renderedWidth / displayedRatio;
            contentOffsetY = (renderedHeight - contentHeight) / 2;
        }
    }

    const scaleX = contentWidth / naturalWidth;
    const scaleY = contentHeight / naturalHeight;
    const left = Math.max(0, Math.min(contentWidth, Number(x1) * scaleX)) + contentOffsetX;
    const top = Math.max(0, Math.min(contentHeight, Number(y1) * scaleY)) + contentOffsetY;
    const width = Math.max(0, Math.min(contentWidth - (left - contentOffsetX), (Number(x2) - Number(x1)) * scaleX));
    const height = Math.max(0, Math.min(contentHeight - (top - contentOffsetY), (Number(y2) - Number(y1)) * scaleY));

    if (stageRect && (videoRect.width === 0 || videoRect.height === 0)) {
        return;
    }

    bboxOverlay.innerHTML = "";
    const rect = document.createElement("div");
    rect.className = "bbox-rect"
    rect.style.left = `${left}px`;
    rect.style.top = `${top}px`;
    rect.style.width = `${width}px`;
    rect.style.height = `${height}px`;

    bboxOverlay.appendChild(rect);
    bboxOverlay.hidden = false;
}

function showBboxOverlay(bbox, label = "bbox") {
    if (!Array.isArray(bbox) || bbox.length !== 4) {
        clearBboxOverlay();
        return;
    }

    activeOverlayBBox = bbox.map((value) => Number(value));
    activeOverlayLabel = label;
    if (videoPlayer.hidden) {
        return;
    }
    refreshBboxOverlay();
}

function setActiveResultItem(activeLi) {
    resultsList.querySelectorAll(".result-item.is-active").forEach((node) => node.classList.remove("is-active"));
    if (activeLi) {
        activeLi.classList.add("is-active");
    }
}

function updateOverlayForVideoSize() {
    if (activeOverlayBBox) {
        refreshBboxOverlay();
    }
}

async function seekToTimestamp(seconds) {
    const ts = toNumberOrNull(seconds);
    if (!videoPlayer || ts == null || !videoPlayer.src) return;

    videoPlayer.hidden = false;

    if (videoPlayer.readyState < 1) {
        await new Promise((resolve) => {
            const onLoaded = () => {
                videoPlayer.removeEventListener("loadedmetadata", onLoaded);
                resolve();
            };
            videoPlayer.addEventListener("loadedmetadata", onLoaded, { once: true });
            videoPlayer.load();
            setTimeout(resolve, 1200);
        });
    }

    const maxTime = Number.isFinite(videoPlayer.duration) ? Math.max(0, videoPlayer.duration - 0.05) : ts;
    videoPlayer.currentTime = Math.min(Math.max(0, ts), maxTime);
    videoPlayer.pause();
}

/**
 * Build a result <li> with a play button and interval info.
 */
function buildResultItem(item) {
    const li = document.createElement("li");
    li.className = "result-item";

    const normalized = normalizeResultItem(item);
    const startTime = normalized.start;
    const endTime = normalized.end;
    const seekTime = normalized.bestTs;
    const score = normalized.score;
    const hitType = normalized.hitType;
    const bbox = normalized.bbox;

    // Play button
    const btn = document.createElement("button");
    btn.className = "result-seek-btn";
    btn.title = `Перейти к ${formatTimestamp(seekTime)}`;
    btn.textContent = "\u25B6";
    btn.addEventListener("click", (e) => {
        e.stopPropagation();
        setActiveResultItem(li);
        seekToTimestamp(seekTime).then(() => showBboxOverlay(bbox, hitType || "bbox"));
    });
    li.appendChild(btn);

    // Text block
    const textDiv = document.createElement("div");
    textDiv.className = "result-text";

    const timeLine = document.createElement("div");
    timeLine.className = "result-time";
    if (startTime != null && endTime != null) {
        timeLine.textContent = `${formatTimeRange(startTime, endTime)} (best: ${formatTimestamp(seekTime)})`;
    } else {
        timeLine.textContent = seekTime != null ? formatTimestamp(seekTime) : "\u2014";
    }
    textDiv.appendChild(timeLine);

    const parts = [];
    if (score != null) parts.push(`score: ${score.toFixed(3)}`);
    if (parts.length) {
        const metaLine = document.createElement("div");
        metaLine.className = "result-meta";
        metaLine.textContent = parts.join(" \u2014 ");
        textDiv.appendChild(metaLine);
    }

    li.appendChild(textDiv);

    li.style.cursor = "pointer";
    li.addEventListener("click", () => {
        setActiveResultItem(li);
        seekToTimestamp(seekTime).then(() => showBboxOverlay(bbox, hitType || "bbox"));
    });

    return li;
}

function setResults(items) {
    clearBboxOverlay();
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

videoPlayer.addEventListener("play", clearBboxOverlay);
videoPlayer.addEventListener("pause", updateOverlayForVideoSize);
videoPlayer.addEventListener("seeked", updateOverlayForVideoSize);
videoPlayer.addEventListener("loadedmetadata", updateOverlayForVideoSize);
window.addEventListener("resize", updateOverlayForVideoSize);

if (bboxToggle) {
    bboxOverlayEnabled = bboxToggle.checked;
    bboxToggle.addEventListener("change", () => {
        bboxOverlayEnabled = bboxToggle.checked;
        if (bboxOverlayEnabled) {
            refreshBboxOverlay();
        } else {
            if (bboxOverlay) {
                bboxOverlay.innerHTML = "";
                bboxOverlay.hidden = true;
            }
        }
    });
}

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
        syncSearchControls();
    } catch (err) {
        setStatus(userStatus, "ошибка", "status-failed");
        syncSearchControls();
        alert(`Ошибка идентификации: ${err.message}`);
    }
});

// ======================== VIDEO ========================

fileInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) {
        videoStateVersion += 1;
        videoId = null;
        videoProcessingStatus = null;
        selectedFile = file;
        const mimeOk = file.type && SUPPORTED_VIDEO_TYPES.includes(file.type);
        if (!mimeOk && file.type) {
            setStatus(videoStatus, `\u26A0 Формат ${file.type} может не воспроизводиться`, "status-pending");
        } else {
            setStatus(videoStatus, "готово к загрузке", "");
        }
        revokeCurrentVideoUrl();
        currentVideoUrl = URL.createObjectURL(file);
        videoPlayer.src = currentVideoUrl;
        if (videoStage) videoStage.hidden = false;
        videoPlayer.hidden = false;
        videoPlayer.load();
        resetSearchUi();
        syncUploadControls();
    }
});

removeBtn.addEventListener("click", () => {
    videoStateVersion += 1;
    videoProcessingStatus = null;
    resetVideoPreview();
    fileInput.value = "";
    selectedFile = null;
    videoId = null;
    setStatus(videoStatus, "не загружено", "");
    resetSearchUi();
    syncUploadControls();
    syncSearchControls();
});

uploadBtn.addEventListener("click", async () => {
    if (isUploadingVideo || videoId !== null) {
        return;
    }
    if (!userId) {
        alert("Сначала введите никнейм.");
        return;
    }
    if (!selectedFile) {
        alert("Выберите видео.");
        return;
    }

    const currentVideoVersion = ++videoStateVersion;
    isUploadingVideo = true;
    videoProcessingStatus = "pending";
    syncUploadControls();
    syncSearchControls();
    setStatus(videoStatus, "загрузка на сервер...", "status-pending");
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("user_id", String(userId));

    try {
        const data = await requestJson(`${API_BASE}/videos`, {
            method: "POST",
            body: formData
        });
        if (currentVideoVersion !== videoStateVersion) return;

        videoId = data.video_id;
        videoProcessingStatus = data.status;
        setStatus(videoStatus, `загружено (id: ${videoId}) \u2014 ${statusLabel(data.status)}`, statusClass(data.status));
        void pollVideoStatus(videoId, currentVideoVersion);
    } catch (err) {
        if (currentVideoVersion !== videoStateVersion) return;

        videoProcessingStatus = null;
        setStatus(videoStatus, "ошибка загрузки", "status-failed");
        alert(`Ошибка загрузки: ${err.message}`);
    } finally {
        if (currentVideoVersion === videoStateVersion) {
            isUploadingVideo = false;
            syncUploadControls();
            syncSearchControls();
        }
    }
});

/**
 * Poll GET /videos/{id} until processing_status is terminal.
 */
async function pollVideoStatus(id, currentVideoVersion) {
    const terminalStatuses = ["ready", "completed", "failed"];
    const maxTries = 120;

    for (let i = 0; i < maxTries; i++) {
        if (currentVideoVersion !== videoStateVersion || videoId !== id) {
            return;
        }
        await new Promise((r) => setTimeout(r, 5000));
        try {
            const data = await requestJson(`${API_BASE}/videos/${id}`, { method: "GET" });
            if (currentVideoVersion !== videoStateVersion || videoId !== id) {
                return;
            }

            const st = data.status;
            videoProcessingStatus = st;
            setStatus(videoStatus, `видео (id: ${id}) \u2014 ${statusLabel(st)}`, statusClass(st));
            syncSearchControls();
            if (terminalStatuses.includes(st)) {
                return;
            }
        } catch (err) {
            console.warn("[PollVideo]", err.message);
        }
    }
    if (currentVideoVersion === videoStateVersion && videoId === id) {
        videoProcessingStatus = "failed";
        setStatus(videoStatus, `видео (id: ${id}) \u2014 таймаут ожидания`, "status-failed");
        syncSearchControls();
    }
}

// ======================== SEARCH ========================

searchForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (isSearching) {
        return;
    }
    if (!userId || !videoId) {
        alert("Нужны пользователь и загруженное видео.");
        return;
    }
    if (!isVideoReadyForSearch()) {
        alert("Дождитесь, пока видео закончит обработку.");
        return;
    }
    const queryText = promptInput.value.trim();
    if (!queryText) return;

    const currentSearchVersion = ++searchStateVersion;
    isSearching = true;
    syncSearchControls();
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
        if (currentSearchVersion !== searchStateVersion) return;

        const queryId = data.query_id;
        setStatus(searchStatus, `в обработке... (id: ${queryId})`, "status-pending");
        await pollSearch(queryId, currentSearchVersion);
    } catch (err) {
        if (currentSearchVersion !== searchStateVersion) return;

        setStatus(searchStatus, "ошибка", "status-failed");
        alert(`Ошибка поиска: ${err.message}`);
    } finally {
        if (currentSearchVersion === searchStateVersion) {
            isSearching = false;
            syncSearchControls();
        }
    }
});

async function pollSearch(queryId, currentSearchVersion) {
    const maxTries = 120;
    const terminalStatuses = ["ready", "completed", "not_found", "failed"];

    for (let i = 0; i < maxTries; i++) {
        if (currentSearchVersion !== searchStateVersion) {
            return;
        }
        try {
            const statusData = await requestJson(`${API_BASE}/searches/${queryId}`, {
                method: "GET"
            });
            if (currentSearchVersion !== searchStateVersion) {
                return;
            }

            const st = statusData.status;
            setStatus(searchStatus, statusLabel(st), statusClass(st));

            if (terminalStatuses.includes(st)) {
                if (st === "ready" || st === "completed") {
                    try {
                        const resultsData = await requestJson(`${API_BASE}/searches/${queryId}/results`, {
                            method: "GET"
                        });
                        if (currentSearchVersion !== searchStateVersion) {
                            return;
                        }

                        const result = Array.isArray(resultsData.result)
                            ? resultsData.result
                            : Array.isArray(resultsData.results)
                                ? resultsData.results
                                : [];
                        setResults(result);
                    } catch (err) {
                        if (currentSearchVersion !== searchStateVersion) {
                            return;
                        }

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
    if (currentSearchVersion === searchStateVersion) {
        setStatus(searchStatus, "таймаут ожидания", "status-failed");
    }
}

syncUploadControls();
syncSearchControls();
