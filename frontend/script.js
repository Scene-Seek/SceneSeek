const fileInput = document.getElementById("fileInput");
const videoPlayer = document.getElementById("videoPlayer")
const removeBtn = document.getElementById("removeBtn");

let currentVideoUrl = null;

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