const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "";

async function checkBackend() {
  const status = document.getElementById("status");

  try {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    status.textContent = `Backend status: ${data.status}`;
  } catch (error) {
    status.textContent = "Backend status: unavailable";
  }
}

async function uploadImage(event) {
  event.preventDefault();

  const status = document.getElementById("status");
  const fileInput = document.getElementById("image-file");
  const file = fileInput.files[0];

  if (!file) {
    status.textContent = "Choose an image first.";
    return;
  }

  const formData = new FormData();
  formData.append("image", file);
  status.textContent = "Uploading image...";

  try {
    const response = await fetch(`${API_BASE}/upload_image`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    const uploaded = await response.json();
    drawImage(uploaded.url, uploaded.width, uploaded.height);
    showImageMeta(uploaded);
    status.textContent = "Image uploaded.";
  } catch (error) {
    status.textContent = `Upload failed: ${error.message}`;
  }
}

function showImageMeta(uploaded) {
  document.getElementById("image-id").textContent = uploaded.image_id;
  document.getElementById("image-width").textContent = uploaded.width;
  document.getElementById("image-height").textContent = uploaded.height;
}

function drawImage(url, width, height) {
  const canvas = document.getElementById("image-canvas");
  const context = canvas.getContext("2d");
  const image = new Image();

  image.onload = () => {
    canvas.width = width;
    canvas.height = height;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0);
  };

  image.src = url;
}

document.getElementById("upload-form").addEventListener("submit", uploadImage);
checkBackend();
