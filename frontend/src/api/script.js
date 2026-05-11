const authData = getStoredAuthData();
let userId = authData.userId;
let username = authData.username;
let accessToken = authData.accessToken;

if (!userId || !accessToken) {
    clearAuthData();
    window.location.href = "index.html";
}

document.getElementById("current-user-name").textContent = username || "Гость";
const userModeBadge = document.getElementById("user-mode-badge");
if (userModeBadge) {
    userModeBadge.hidden = !authData.isAnonymous;
}

document.getElementById("logout-btn").addEventListener("click", () => {
    clearAuthData();
    window.location.href = "index.html";
});

const fileInput = document.getElementById("fileInput");
const videoPlayer = document.getElementById("videoPlayer");
const removeBtn = document.getElementById("removeBtn");
const uploadBtn = document.getElementById("uploadBtn");

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
const historyList = document.getElementById("history-list");
const historyStatus = document.getElementById("history-status");
const refreshHistoryBtn = document.getElementById("refresh-history-btn");

let currentVideoUrl = null;
let selectedFile = null;
let videoId = null;
let videoProcessingStatus = null;
let isUploadingVideo = false;
let isSearching = false;
let videoStateVersion = 0;
let searchStateVersion = 0;
let activeOverlayBBox = null;
let activeOverlayLabel = "";
let bboxOverlayEnabled = true;

if (videoStage) videoStage.hidden = true;
if (videoPlayer) videoPlayer.hidden = true;

const SUPPORTED_VIDEO_TYPES = [
    "video/mp4", "video/webm", "video/ogg",
    "video/quicktime", "video/x-m4v", "application/mp4"
];

const STATUS_LABELS = {
    pending: "Ожидание",
    indexing: "Индексация...",
    ready: "Готово",
    completed: "Завершено",
    not_found: "Ничего не найдено",
    failed: "Ошибка",
};
const VIDEO_READY_STATUSES = new Set(["ready", "completed"]);
const VIDEO_TERMINAL_STATUSES = new Set(["ready", "completed", "failed"]);
const VIDEO_REMOVABLE_STATUSES = new Set(["ready", "completed", "failed"]);
const SEARCH_TERMINAL_STATUSES = new Set(["ready", "completed", "not_found", "failed"]);

/**
 * Сохраняет состояние приложения в localStorage
 */
function saveAppState() {
    // Сохранить текущее состояние приложения в localStorage
    if (!userId) return;
    
    const state = {
        videoId: videoId,
        videoProcessingStatus: videoProcessingStatus,
        prompt: promptInput.value
    };
    localStorage.setItem(`app_state_${userId}`, JSON.stringify(state));
}

/**
 * Загружает сохраненное состояние приложения из localStorage
 */
function loadAppState() {
    // Загрузить сохраненное состояние приложения из localStorage
    if (!userId) return;
    
    const saved = localStorage.getItem(`app_state_${userId}`);
    if (!saved) return;

    const state = JSON.parse(saved);

    if (state.prompt) {
        promptInput.value = state.prompt;
    }

    if (state.videoId) {
        videoId = state.videoId;
        videoProcessingStatus = state.videoProcessingStatus;
        setStatus(videoStatus, `${statusLabel(videoProcessingStatus)}`, statusClass(videoProcessingStatus));
        void restoreSavedVideo(videoId, ++videoStateVersion);
    }
    
    syncSearchControls();
    syncUploadControls();
}

/**
 * Восстанавливает видео по идентификатору и синхронизирует UI
 */
async function restoreSavedVideo(id, currentVideoVersion) {
    try {
        const data = await requestJson(`${API_BASE}/videos/${id}`, { method: "GET" });
        if (currentVideoVersion !== videoStateVersion || videoId !== id) return;

        videoId = data.video_id;
        videoProcessingStatus = data.status;
        videoPlayer.src = data.video_path;
        videoPlayer.hidden = false;
        if (videoStage) videoStage.hidden = false;
        videoPlayer.load();
        setStatus(videoStatus, `${statusLabel(videoProcessingStatus)}`, statusClass(videoProcessingStatus));

        if (!VIDEO_TERMINAL_STATUSES.has(videoProcessingStatus)) {
            void pollVideoStatus(videoId, currentVideoVersion);
        }
    } catch (err) {
        if (currentVideoVersion !== videoStateVersion) return;
        if (!localStorage.getItem("accessToken")) return;
        videoId = null;
        videoProcessingStatus = null;
        resetVideoPreview();
        setStatus(videoStatus, "не удалось восстановить видео", "status-failed");
        saveAppState();
    } finally {
        if (currentVideoVersion === videoStateVersion) {
            syncUploadControls();
            syncSearchControls();
        }
    }
}

promptInput.addEventListener('input', saveAppState);

/**
 * Возвращает читаемую подпись для статуса
 */
function statusLabel(raw) { return STATUS_LABELS[raw] || raw; }

/**
 * Устанавливает текст и класс статуса на элементе
 */
function setStatus(el, text, cssClass) {
    el.textContent = text;
    el.className = "";
    if (cssClass) el.classList.add(cssClass);
}

/**
 * Определяет CSS класс по статусу
 */
function statusClass(raw) {
    if (raw === "ready" || raw === "completed") return "status-ready";
    if (raw === "pending" || raw === "indexing") return "status-pending";
    if (raw === "failed" || raw === "error") return "status-failed";
    return "";
}

/**
 * Проверяет готовность видео для поиска
 */
function isVideoReadyForSearch() {
    return VIDEO_READY_STATUSES.has(videoProcessingStatus);
}

/**
 * Проверяет возможность удаления текущего видео
 */
function canRemoveCurrentVideo() {
    if (isUploadingVideo) return false;
    if (videoId === null) return true;
    return VIDEO_REMOVABLE_STATUSES.has(videoProcessingStatus);
}

/**
 * Синхронизирует состояние контролов загрузки
 */
function syncUploadControls() {
    const isLocked = isUploadingVideo || videoId !== null;
    const hasVideo = Boolean(selectedFile || videoId || currentVideoUrl);
    fileInput.disabled = isLocked;
    uploadBtn.disabled = !selectedFile || isLocked;
    removeBtn.hidden = !hasVideo;
    removeBtn.disabled = !hasVideo || !canRemoveCurrentVideo();
    removeBtn.title = removeBtn.disabled && hasVideo
        ? "Дождитесь завершения индексации"
        : "";
    if (dropArea) {
        dropArea.classList.toggle("is-disabled", isLocked);
        dropArea.setAttribute("aria-disabled", String(isLocked));
    }
}

/**
 * Синхронизирует состояние контролов поиска
 */
function syncSearchControls() {
    searchSubmitBtn.disabled = isSearching || !userId || !videoId || !isVideoReadyForSearch();
}

/**
 * Освобождает текущий object URL видео
 */
function revokeCurrentVideoUrl() {
    if (currentVideoUrl) {
        URL.revokeObjectURL(currentVideoUrl);
        currentVideoUrl = null;
    }
}

/**
 * Сбрасывает превью видео и оверлей
 */
function resetVideoPreview() {
    videoPlayer.pause();
    videoPlayer.removeAttribute("src");
    videoPlayer.load();
    videoPlayer.hidden = true;
    if (videoStage) videoStage.hidden = true;
    resetBboxOverlayState();
    revokeCurrentVideoUrl();
}

/**
 * Сбрасывает интерфейс поиска и результаты
 */
function resetSearchUi() {
    searchStateVersion += 1;
    isSearching = false;
    setStatus(searchStatus, "ожидание", "");
    resetBboxOverlayState();
    resultsList.innerHTML = "";
    syncSearchControls();
}

/**
 * Форматирует секунды в строку времени
 */
function formatTimestamp(seconds) {
    if (seconds == null || Number.isNaN(Number(seconds))) return "??:??";
    const totalSec = Math.floor(Number(seconds));
    const h = Math.floor(totalSec / 3600);
    const m = Math.floor((totalSec % 3600) / 60);
    const s = totalSec % 60;
    if (h > 0) return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

/**
 * Форматирует временной диапазон в человекочитаемый вид
 */
function formatTimeRange(startSec, endSec) {
    return `${formatTimestamp(startSec)} – ${formatTimestamp(endSec)}`;
}

/**
 * Преобразует значение в число или null
 */
function toNumberOrNull(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
}

/**
 * Нормализует bbox в массив чисел
 */
function normalizeBbox(value) {
    if (Array.isArray(value)) return value.map(Number).filter(Number.isFinite);
    if (typeof value === "string") {
        try {
            const parsed = JSON.parse(value);
            if (Array.isArray(parsed)) return parsed.map(Number).filter(Number.isFinite);
        } catch { return []; }
    }
    return [];
}

/**
 * Нормализует элемент результата поиска
 */
function normalizeResultItem(item) {
    const start = toNumberOrNull(item.start ?? item.start_time ?? item.segment_start);
    const end = toNumberOrNull(item.end ?? item.end_time ?? item.segment_end);
    const bestTs = toNumberOrNull(item.best_ts ?? item.bestTs ?? item.timestamp ?? item.time ?? item.t ?? start);
    const score = toNumberOrNull(item.score ?? item.similarity_score);
    const hitType = item.type ?? item.hit_type ?? item?.yolo_metadata?.type ?? item?.metadata?.type;
    const bbox = normalizeBbox(item.bbox ?? item?.yolo_metadata?.bbox ?? item?.metadata?.bbox);
    return { start, end, bestTs, score, hitType, bbox };
}

/**
 * Очищает слой bbox
 */
function clearBboxOverlay() {
    if (!bboxOverlay) return;
    bboxOverlay.innerHTML = "";
    bboxOverlay.hidden = true;
}

/**
 * Сбрасывает состояние bbox оверлея
 */
function resetBboxOverlayState() {
    activeOverlayBBox = null;
    activeOverlayLabel = "";
    clearBboxOverlay();
}

/**
 * Обновляет bbox оверлей с учетом переключателя
 */
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

/**
 * Рисует bbox поверх видео с учетом масштаба
 */
function renderBboxOverlay() {
    if (!bboxOverlayEnabled || !bboxOverlay || !videoPlayer || !activeOverlayBBox || activeOverlayBBox.length !== 4) return;
    const [x1, y1, x2, y2] = activeOverlayBBox;
    const naturalWidth = Number(videoPlayer.videoWidth);
    const naturalHeight = Number(videoPlayer.videoHeight);
    const renderedWidth = videoPlayer.clientWidth;
    const renderedHeight = videoPlayer.clientHeight;
    if (!naturalWidth || !naturalHeight || !renderedWidth || !renderedHeight) return;

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

    if (stageRect && (videoRect.width === 0 || videoRect.height === 0)) return;

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

/**
 * Показывает bbox и метку на текущем кадре
 */
function showBboxOverlay(bbox, label = "bbox") {
    if (!Array.isArray(bbox) || bbox.length !== 4) {
        clearBboxOverlay();
        return;
    }
    activeOverlayBBox = bbox.map(Number);
    activeOverlayLabel = label;
    if (videoPlayer.hidden) return;
    refreshBboxOverlay();
}

/**
 * Делает элемент результата активным
 */
function setActiveResultItem(activeLi) {
    resultsList.querySelectorAll(".result-item.is-active").forEach((node) => node.classList.remove("is-active"));
    if (activeLi) activeLi.classList.add("is-active");
}

/**
 * Перерисовывает bbox при изменении размеров плеера
 */
function updateOverlayForVideoSize() {
    if (activeOverlayBBox) refreshBboxOverlay();
}

/**
 * Перематывает видео на указанную временную метку
 */
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
 * Создает DOM элемент результата поиска
 */
function buildResultItem(item) {
    const li = document.createElement("li");
    li.className = "result-item";
    const normalized = normalizeResultItem(item);
    
    const btn = document.createElement("button");
    btn.className = "result-seek-btn";
    btn.title = `Перейти к ${formatTimestamp(normalized.bestTs)}`;
    btn.textContent = "\u25B6";
    btn.addEventListener("click", (e) => {
        e.stopPropagation();
        setActiveResultItem(li);
        seekToTimestamp(normalized.bestTs).then(() => showBboxOverlay(normalized.bbox, normalized.hitType || "bbox"));
    });
    li.appendChild(btn);

    const textDiv = document.createElement("div");
    textDiv.className = "result-text";
    const timeLine = document.createElement("div");
    timeLine.className = "result-time";
    if (normalized.start != null && normalized.end != null) {
        timeLine.textContent = `${formatTimeRange(normalized.start, normalized.end)} (best: ${formatTimestamp(normalized.bestTs)})`;
    } else {
        timeLine.textContent = normalized.bestTs != null ? formatTimestamp(normalized.bestTs) : "";
    }
    textDiv.appendChild(timeLine);

    const parts = [];
    if (normalized.score != null) parts.push(`score: ${normalized.score.toFixed(3)}`);
    if (parts.length) {
        const metaLine = document.createElement("div");
        metaLine.className = "result-meta";
        metaLine.textContent = parts.join("  ");
        textDiv.appendChild(metaLine);
    }
    li.appendChild(textDiv);

    li.style.cursor = "pointer";
    li.addEventListener("click", () => {
        setActiveResultItem(li);
        seekToTimestamp(normalized.bestTs).then(() => showBboxOverlay(normalized.bbox, normalized.hitType || "bbox"));
    });

    return li;
}

/**
 * Заполняет список результатов поиска
 */
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

/**
 * Обновляет статус блока истории
 */
function setHistoryStatus(text, cssClass = "") {
    if (!historyStatus) return;
    historyStatus.textContent = text;
    historyStatus.className = "";
    if (cssClass) historyStatus.classList.add(cssClass);
}

/**
 * Форматирует дату для истории запросов
 */
function formatHistoryDate(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    return date.toLocaleString("ru-RU", {
        day: "2-digit",
        month: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
    });
}

/**
 * Заполняет список истории
 */
function setHistoryItems(items) {
    if (!historyList) return;
    historyList.innerHTML = "";

    if (!items || items.length === 0) {
        const li = document.createElement("li");
        li.className = "history-empty";
        li.textContent = "История пуста";
        historyList.appendChild(li);
        return;
    }

    items.forEach((item) => historyList.appendChild(buildHistoryItem(item)));
}

/**
 * Создает DOM элемент записи истории
 */
function buildHistoryItem(item) {
    const li = document.createElement("li");
    li.className = "history-item";

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "history-item-btn";

    const query = document.createElement("span");
    query.className = "history-query";
    query.textContent = item.video_title || `video ${item.video_id}`;
    btn.appendChild(query);

    const meta = document.createElement("span");
    meta.className = "history-meta";
    const latestPrompt = item.latest_query_text || "без запросов";
    const latestStatus = item.latest_search_status
        ? statusLabel(item.latest_search_status)
        : statusLabel(item.video_status);
    const date = item.latest_search_date || item.created_at;
    meta.textContent = `${latestPrompt} · ${latestStatus} · ${formatHistoryDate(date)}`;
    btn.appendChild(meta);

    btn.addEventListener("click", () => {
        void openHistoryItem(item);
    });

    li.appendChild(btn);
    return li;
}

/**
 * Загружает историю видео и запросов
 */
async function loadSearchHistory() {
    if (!historyList) return;
    setHistoryStatus("загрузка...", "status-pending");

    try {
        const data = await requestJson(`${API_BASE}/videos/history`, { method: "GET" });
        const history = Array.isArray(data.history) ? data.history : [];
        setHistoryItems(history);
        setHistoryStatus(history.length ? `${history.length}` : "пусто", history.length ? "status-ready" : "");
    } catch (err) {
        setHistoryItems([]);
        setHistoryStatus("ошибка", "status-failed");
    }
}

/**
 * Проверяет возможность открыть запись истории
 */
function canOpenHistoryItem(item) {
    if (isUploadingVideo) return false;
    if (videoId === null) return true;
    if (Number(videoId) === Number(item.video_id)) return true;
    return canRemoveCurrentVideo();
}

/**
 * Загружает данные из истории и обновляет интерфейс
 */
async function openHistoryItem(item) {
    if (!canOpenHistoryItem(item)) {
        alert("Дождитесь завершения индексации текущего видео.");
        return;
    }

    const currentVideoVersion = ++videoStateVersion;
    const currentSearchVersion = ++searchStateVersion;
    isSearching = false;
    selectedFile = null;
    fileInput.value = "";
    revokeCurrentVideoUrl();
    resetBboxOverlayState();
    resultsList.innerHTML = "";

    videoId = item.video_id;
    videoProcessingStatus = item.video_status;
    promptInput.value = item.latest_query_text || "";
    setStatus(videoStatus, `${statusLabel(videoProcessingStatus)}`, statusClass(videoProcessingStatus));
    if (item.latest_search_status) {
        setStatus(searchStatus, statusLabel(item.latest_search_status), statusClass(item.latest_search_status));
    } else {
        setStatus(searchStatus, "ожидание", "");
    }
    saveAppState();
    syncUploadControls();
    syncSearchControls();

    await restoreSavedVideo(videoId, currentVideoVersion);
    if (currentVideoVersion !== videoStateVersion || currentSearchVersion !== searchStateVersion) return;

    if (!item.latest_query_id) {
        setResults([]);
        return;
    }

    if (item.latest_search_status === "ready" || item.latest_search_status === "completed") {
        try {
            const resultsData = await requestJson(`${API_BASE}/searches/${item.latest_query_id}/results`, { method: "GET" });
            if (currentSearchVersion !== searchStateVersion) return;
            const result = Array.isArray(resultsData.result) ? resultsData.result : [];
            setResults(result);
        } catch (err) {
            if (currentSearchVersion !== searchStateVersion) return;
            setStatus(searchStatus, "ошибка получения результатов", "status-failed");
        }
        return;
    }

    if (item.latest_search_status === "not_found" || item.latest_search_status === "failed") {
        setResults([]);
        return;
    }

    if (!SEARCH_TERMINAL_STATUSES.has(item.latest_search_status)) {
        await pollSearch(item.latest_query_id, currentSearchVersion);
    }
}

videoPlayer.addEventListener("error", () => {
    const err = videoPlayer.error;
    if (err) {
        const messages = {
            1: "Воспроизведение прервано",
            2: "Сетевая ошибка загрузки видео",
            3: "Ошибка декодирования видео  формат не поддерживается браузером",
            4: "Формат видео не поддерживается",
        };
        const msg = messages[err.code] || `Ошибка видеоплеера (код ${err.code})`;
        setStatus(videoStatus, msg, "status-failed");
    }
});

videoPlayer.addEventListener("loadedmetadata", () => {
    if (selectedFile && !isUploadingVideo && videoId === null) {
        setStatus(videoStatus, "готово к загрузке", "");
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

fileInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) {
        videoStateVersion += 1;
        videoId = null;
        videoProcessingStatus = null;
        selectedFile = file;
        const mimeOk = file.type && SUPPORTED_VIDEO_TYPES.includes(file.type);
        if (!mimeOk && file.type) {
            setStatus(videoStatus, `формат ${file.type} может не воспроизводиться`, "status-pending");
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
        saveAppState();
    }
});

removeBtn.addEventListener("click", () => {
    if (!canRemoveCurrentVideo()) {
        alert("Дождитесь завершения индексации видео.");
        return;
    }

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
    saveAppState();
});

uploadBtn.addEventListener("click", async () => {
    if (isUploadingVideo || videoId !== null) return;
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

    try {
        const data = await requestJson(`${API_BASE}/videos`, {
            method: "POST",
            body: formData
        });
        if (currentVideoVersion !== videoStateVersion) return;

        videoId = data.video_id;
        videoProcessingStatus = data.status;
        setStatus(videoStatus, ` ${statusLabel(data.status)}`, statusClass(data.status));
        void pollVideoStatus(videoId, currentVideoVersion);
        saveAppState();
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
 * Опрашивает статус обработки видео до завершения
 */
async function pollVideoStatus(id, currentVideoVersion) {
    const maxTries = 120;
    for (let i = 0; i < maxTries; i++) {
        if (currentVideoVersion !== videoStateVersion || videoId !== id) return;
        await new Promise((r) => setTimeout(r, 5000));
        try {
            const data = await requestJson(`${API_BASE}/videos/${id}`, { method: "GET" });
            if (currentVideoVersion !== videoStateVersion || videoId !== id) return;
            const st = data.status;
            videoProcessingStatus = st;
            saveAppState();
            setStatus(videoStatus, ` ${statusLabel(st)}`, statusClass(st));
            syncUploadControls();
            syncSearchControls();
            if (VIDEO_TERMINAL_STATUSES.has(st)) return;
        } catch (err) {
            console.warn("[PollVideo]", err.message);
        }
    }
    if (currentVideoVersion === videoStateVersion && videoId === id) {
        videoProcessingStatus = "failed";
        setStatus(videoStatus, ` таймаут ожидания`, "status-failed");
        syncUploadControls();
        syncSearchControls();
    }
}

searchForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (isSearching) return;
    if (!videoId) {
        alert("Нужно загруженное видео.");
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
                video_id: videoId,
                query_text: queryText
            })
        });
        saveAppState();
        if (currentSearchVersion !== searchStateVersion) return;
        const queryId = data.query_id;
        void loadSearchHistory();
        setStatus(searchStatus, `в обработке... `, "status-pending");
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

    for (let i = 0; i < maxTries; i++) {
        if (currentSearchVersion !== searchStateVersion) return;
        try {
            const statusData = await requestJson(`${API_BASE}/searches/${queryId}`, { method: "GET" });
            if (currentSearchVersion !== searchStateVersion) return;
            const st = statusData.status;
            setStatus(searchStatus, statusLabel(st), statusClass(st));

            if (SEARCH_TERMINAL_STATUSES.has(st)) {
                if (st === "ready" || st === "completed") {
                    try {
                        const resultsData = await requestJson(`${API_BASE}/searches/${queryId}/results`, { method: "GET" });
                        if (currentSearchVersion !== searchStateVersion) return;
                        const result = Array.isArray(resultsData.result) ? resultsData.result : Array.isArray(resultsData.results) ? resultsData.results : [];
                        setResults(result);
                    } catch (err) {
                        if (currentSearchVersion !== searchStateVersion) return;
                        setResults([]);
                        setStatus(searchStatus, "ошибка получения результатов", "status-failed");
                    }
                } else if (st === "not_found") {
                    setResults([]);
                } else if (st === "failed") {
                    setResults([]);
                    setStatus(searchStatus, "ошибка обработки", "status-failed");
                }
                void loadSearchHistory();
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

if (refreshHistoryBtn) {
    refreshHistoryBtn.addEventListener("click", () => {
        void loadSearchHistory();
    });
}

window.addEventListener("load", () => {
    loadAppState();
    void loadSearchHistory();
});
syncUploadControls();
syncSearchControls();
