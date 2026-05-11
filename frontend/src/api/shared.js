const API_BASE = (() => {
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    const apiPort = 8000;
    return `${protocol}//${hostname}:${apiPort}/api/v1`;
})();

/**
 * Читает данные авторизации из localStorage
 */
function getStoredAuthData() {
    return {
        userId: localStorage.getItem("userId"),
        username: localStorage.getItem("username") || localStorage.getItem("nickname"),
        accessToken: localStorage.getItem("accessToken"),
        isAnonymous: localStorage.getItem("isAnonymous") === "true",
    };
}

/**
 * Сохраняет данные авторизации в localStorage
 */
function storeAuthData(data) {
    localStorage.setItem("userId", String(data.user_id));
    localStorage.setItem("username", data.username);
    localStorage.setItem("accessToken", data.token);
    localStorage.setItem("isAnonymous", String(Boolean(data.is_anonymous)));
    localStorage.removeItem("nickname");
}

/**
 * Очищает данные авторизации и состояние приложения
 */
function clearAuthData() {
    const userId = localStorage.getItem("userId");
    if (userId) {
        localStorage.removeItem(`app_state_${userId}`);
    }
    localStorage.removeItem("userId");
    localStorage.removeItem("username");
    localStorage.removeItem("nickname");
    localStorage.removeItem("accessToken");
    localStorage.removeItem("isAnonymous");
}

/**
 * Перенаправляет на страницу входа при отсутствии сессии
 */
function redirectToLogin() {
    if (!window.location.pathname.endsWith("/index.html") && window.location.pathname !== "/") {
        window.location.href = "index.html";
    }
}

/**
 * Формирует читаемое сообщение об ошибке из ответа API
 */
function errorMessageFromPayload(payload, fallback) {
    if (!payload) return fallback;
    if (typeof payload === "string") return payload;
    if (typeof payload.detail === "string") return payload.detail;
    if (Array.isArray(payload.detail)) {
        return payload.detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
    }
    return fallback;
}

/**
 * Делает запрос к API и возвращает JSON с учетом авторизации
 */
async function requestJson(url, options = {}) {
    const { auth = true, headers, ...fetchOptions } = options;
    const requestHeaders = new Headers(headers || {});
    const token = localStorage.getItem("accessToken");

    if (auth && token) {
        requestHeaders.set("Authorization", `Bearer ${token}`);
    }

    const response = await fetch(url, {
        ...fetchOptions,
        headers: requestHeaders,
    });

    if (!response.ok) {
        let payload = null;
        const text = await response.text();
        try {
            payload = text ? JSON.parse(text) : null;
        } catch {
            payload = text;
        }

        if (response.status === 401 && auth) {
            clearAuthData();
            redirectToLogin();
        }

        throw new Error(errorMessageFromPayload(payload, `HTTP ${response.status}`));
    }

    if (response.status === 204) return null;
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) return null;
    return response.json();
}
