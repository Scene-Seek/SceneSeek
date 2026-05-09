if (localStorage.getItem("accessToken") && localStorage.getItem("userId")) {
    window.location.href = 'main.html';
}

const identifyForm = document.getElementById("identify-form");
const nicknameInput = document.getElementById("nickname-input");
const passwordInput = document.getElementById("password-input");
const userStatus = document.getElementById("user-status");

identifyForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const nickname = nicknameInput.value.trim();
    const password = passwordInput.value;
    if (!nickname || !password) return;
    
    userStatus.textContent = "идёт запрос...";
    userStatus.className = "status-pending";
    
    try {
        const data = await requestJson(`${API_BASE}/identify`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ nickname, password }),
            auth: false,
        });
        
        localStorage.setItem("userId", data.user_id);
        localStorage.setItem("nickname", data.nickname);
        localStorage.setItem("accessToken", data.token);
        
        userStatus.textContent = "Успешно! Перенаправление...";
        userStatus.className = "status-ready";
        
        window.location.href = 'main.html';
    } catch (err) {
        userStatus.textContent = "Ошибка";
        userStatus.className = "status-failed";
        alert(`Ошибка идентификации: ${err.message}`);
    }
});
