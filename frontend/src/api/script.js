const API_BASE = "http://127.0.0.1:8000/api/v1";

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

function setStatus(el, text) {
    el.textContent = text;
}

function formatResult(item) {
    if (typeof item === "number" || typeof item === "string") {
        return String(item);
    }
    if (item && typeof item === "object") {
        const timestamp = item.timestamp ?? item.time ?? item.t;
        const score = item.score ?? item.similarity_score;
        if (timestamp !== undefined && score !== undefined) {
            return `t=${timestamp} score=${score}`;
        }
        if (timestamp !== undefined) {
            return `t=${timestamp}`;
        }
        return JSON.stringify(item);
    }
    return String(item);
}

function setResults(items) {
    resultsList.innerHTML = "";
    if (!items || items.length === 0) {
        const li = document.createElement("li");
        li.textContent = "Ничего не найдено";
        resultsList.appendChild(li);
        return;
    }
    items.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = formatResult(item);
        resultsList.appendChild(li);
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

identifyForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const nickname = nicknameInput.value.trim();
    if (!nickname) return;
    setStatus(userStatus, "идёт запрос...");
    try {
        const data = await requestJson(`${API_BASE}/identify`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ nickname })
        });
        userId = data.user_id;
        setStatus(userStatus, `${data.nickname} (id: ${userId})`);
    } catch (err) {
        setStatus(userStatus, "ошибка");
        alert(`Ошибка идентификации: ${err.message}`);
    }
});

fileInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) {
        selectedFile = file;
        currentVideoUrl = URL.createObjectURL(file);
        videoPlayer.src = currentVideoUrl;
        videoPlayer.hidden = false;
        videoPlayer.play();
        removeBtn.hidden = false;
        setStatus(videoStatus, "готово к загрузке");
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
    setStatus(videoStatus, "не загружено");
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
    setStatus(videoStatus, "загрузка...");
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("user_id", String(userId));
    try {
        const data = await requestJson(`${API_BASE}/videos`, {
            method: "POST",
            body: formData
        });
        videoId = data.video_id;
        setStatus(videoStatus, `загружено (id: ${videoId}, статус: ${data.status})`);
    } catch (err) {
        setStatus(videoStatus, "ошибка");
        alert(`Ошибка загрузки: ${err.message}`);
    }
});

searchForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!userId || !videoId) {
        alert("Нужны пользователь и загруженное видео.");
        return;
    }
    const queryText = promptInput.value.trim();
    if (!queryText) return;
    setStatus(searchStatus, "отправка...");
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
        setStatus(searchStatus, "в обработке...");
        await pollSearch(queryId);
    } catch (err) {
        setStatus(searchStatus, "ошибка");
        alert(`Ошибка поиска: ${err.message}`);
    }
});

async function pollSearch(queryId) {
    const maxTries = 60;
    for (let i = 0; i < maxTries; i += 1) {
        const statusData = await requestJson(`${API_BASE}/searches/${queryId}`, {
            method: "GET"
        });
        setStatus(searchStatus, statusData.status);

        try {
            const resultsData = await requestJson(`${API_BASE}/searches/${queryId}/results`, {
                method: "GET"
            });
            const result = Array.isArray(resultsData.result)
                ? resultsData.result
                : Array.isArray(resultsData.results)
                    ? resultsData.results
                    : Array.isArray(resultsData)
                        ? resultsData
                        : [];
            if (result.length > 0) {
                setResults(result);
                return;
            }
        } catch (err) {
            setResults([]);
            setStatus(searchStatus, "ошибка результатов");
            alert(`Ошибка получения результатов: ${err.message}`);
            return;
        }

        if (["ready", "found", "not_found", "failed"].includes(statusData.status)) {
            break;
        }
        await new Promise((resolve) => setTimeout(resolve, 2000));
    }
    setStatus(searchStatus, "нет результатов");
}