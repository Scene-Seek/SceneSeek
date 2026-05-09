const API_BASE = (() => {
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    const apiPort = 8000;
    return `${protocol}//${hostname}:${apiPort}/api/v1`;
})();

function clearAuthData() {
    const userId = localStorage.getItem("userId");
    if (userId) {
        localStorage.removeItem(`app_state_${userId}`);
    }
    localStorage.removeItem("userId");
    localStorage.removeItem("nickname");
    localStorage.removeItem("accessToken");
}

function redirectToLogin() {
    if (!window.location.pathname.endsWith("/index.html") && window.location.pathname !== "/") {
        window.location.href = "index.html";
    }
}

function errorMessageFromPayload(payload, fallback) {
    if (!payload) return fallback;
    if (typeof payload === "string") return payload;
    if (typeof payload.detail === "string") return payload.detail;
    if (Array.isArray(payload.detail)) {
        return payload.detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
    }
    return fallback;
}

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
