const fileInput = document.getElementById("fileInput");
const videoPlayer = document.getElementById("videoPlayer")
const removeBtn = document.getElementById("removeBtn");
const userForm = document.getElementById("user-form");
const userInput = document.getElementById("user-input");
const chatHistoryArea = document.getElementById("chat-history-area");

let currentVideoUrl = null;

// Загрузить файл
fileInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    console.log(`INFO: user chose file: ${file.name}`);
    if (file) {
        currentVideoUrl = URL.createObjectURL(file);
        videoPlayer.src = currentVideoUrl;
        videoPlayer.play();
        videoPlayer.hidden = false;
        removeBtn.hidden = false;
    }
});

response = fetch()

// Удалить видео
removeBtn.addEventListener("click", (event) => {
    console.log("INFO: user deleted file");
    videoPlayer.pause();
    videoPlayer.src = ""; 
    videoPlayer.hidden = true;
    removeBtn.hidden = true;
    if (currentVideoUrl) {
        URL.revokeObjectURL(currentVideoUrl);
        currentVideoUrl = null;
    }
});

// Меняет div элемент в чате с ИИ
function addMessage(text, sender) {
    const msgDiv = document.createElement('div');
    msgDiv.textContent = text;
    msgDiv.classList.add('message', sender);
    chatHistoryArea.appendChild(msgDiv);
    chatHistoryArea.scrollTop = chatHistoryArea.scrollHeight;
}

// Отправить сообшение
userForm.addEventListener('submit', (event) => {
    event.preventDefault();    
    const text = userInput.value;
    if (!text) return;
    addMessage(text, 'user');
    userInput.value = '';
    const loadingDiv = document.createElement('div');
    loadingDiv.textContent = "Typing...";
    loadingDiv.classList.add('message', 'ai');
    chatHistoryArea.appendChild(loadingDiv);
    setTimeout(() => {
        loadingDiv.remove();
        addMessage("I received: " + text, 'ai');
    }, 1000);
});