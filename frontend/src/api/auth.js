if (localStorage.getItem("accessToken") && localStorage.getItem("userId")) {
    window.location.href = "main.html";
}

const authForm = document.getElementById("auth-form");
const usernameInput = document.getElementById("username-input");
const passwordInput = document.getElementById("password-input");
const authStatus = document.getElementById("auth-status");
const authSubmitBtn = document.getElementById("auth-submit-btn");
const anonymousBtn = document.getElementById("anonymous-btn");
const loginModeBtn = document.getElementById("login-mode-btn");
const registerModeBtn = document.getElementById("register-mode-btn");

let authMode = "login";

/**
 * Обновляет текст и стиль статуса авторизации
 */
function setAuthStatus(text, cssClass = "") {
    authStatus.textContent = text;
    authStatus.className = cssClass;
}

/**
 * Переключает режим между входом и регистрацией
 */
function setAuthMode(mode) {
    authMode = mode;
    const isLogin = authMode === "login";

    loginModeBtn.classList.toggle("is-active", isLogin);
    registerModeBtn.classList.toggle("is-active", !isLogin);
    loginModeBtn.setAttribute("aria-pressed", String(isLogin));
    registerModeBtn.setAttribute("aria-pressed", String(!isLogin));
    authSubmitBtn.textContent = isLogin ? "Войти" : "Зарегистрироваться";
    passwordInput.autocomplete = isLogin ? "current-password" : "new-password";
    setAuthStatus(isLogin ? "ожидание входа" : "ожидание регистрации");
}

/**
 * Выполняет запрос авторизации и сохраняет данные пользователя
 */
async function authorize(endpoint, payload) {
    const options = {
        method: "POST",
        auth: false,
    };
    if (payload) {
        options.headers = { "Content-Type": "application/json" };
        options.body = JSON.stringify(payload);
    }

    const data = await requestJson(`${API_BASE}${endpoint}`, options);

    storeAuthData(data);
    window.location.href = "main.html";
}

loginModeBtn.addEventListener("click", () => setAuthMode("login"));
registerModeBtn.addEventListener("click", () => setAuthMode("register"));

authForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const username = usernameInput.value.trim();
    const password = passwordInput.value;
    if (!username || !password) return;

    const isLogin = authMode === "login";
    setAuthStatus(isLogin ? "выполняю вход..." : "создаю аккаунт...", "status-pending");
    authSubmitBtn.disabled = true;
    anonymousBtn.disabled = true;

    try {
        await authorize(isLogin ? "/auth/login" : "/auth/register", { username, password });
    } catch (err) {
        setAuthStatus(err.message, "status-failed");
    } finally {
        authSubmitBtn.disabled = false;
        anonymousBtn.disabled = false;
    }
});

anonymousBtn.addEventListener("click", async () => {
    setAuthStatus("создаю гостевую сессию...", "status-pending");
    authSubmitBtn.disabled = true;
    anonymousBtn.disabled = true;

    try {
        await authorize("/auth/anonymous");
    } catch (err) {
        setAuthStatus(err.message, "status-failed");
    } finally {
        authSubmitBtn.disabled = false;
        anonymousBtn.disabled = false;
    }
});

setAuthMode("login");
