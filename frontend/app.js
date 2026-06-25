let apiBasePromise = null;
const canvas = document.getElementById("image-canvas");
const context = canvas.getContext("2d");
const promptJson = document.getElementById("prompt-json");

const state = {
  activeTool: "box",
  image: null,
  imageMeta: null,
  points: [],
  pointLabels: [],
  box: null,
  draftBox: null,
  isDrawingBox: false,
};

async function getApiBase() {
  if (!apiBasePromise) {
    apiBasePromise = resolveApiBase();
  }

  return apiBasePromise;
}

async function resolveApiBase() {
  if (window.location.protocol !== "file:") {
    return "";
  }

  for (let port = 8000; port <= 8010; port += 1) {
    const candidate = `http://127.0.0.1:${port}`;

    try {
      const response = await fetch(`${candidate}/health`);
      if (response.ok) {
        return candidate;
      }
    } catch (error) {
      // Try the next local development port.
    }
  }

  return "http://127.0.0.1:8000";
}

async function checkBackend() {
  const status = document.getElementById("status");

  try {
    const apiBase = await getApiBase();
    const response = await fetch(`${apiBase}/health`);
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
    const apiBase = await getApiBase();
    const response = await fetch(`${apiBase}/upload_image`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    const uploaded = await response.json();
    loadImage(uploaded.url, uploaded);
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

function loadImage(url, uploaded) {
  const image = new Image();
  image.crossOrigin = "anonymous";

  image.onload = () => {
    state.image = image;
    state.imageMeta = {
      image_id: uploaded.image_id,
      width: uploaded.width,
      height: uploaded.height,
    };
    state.points = [];
    state.pointLabels = [];
    state.box = null;
    state.draftBox = null;
    state.isDrawingBox = false;

    canvas.width = uploaded.width;
    canvas.height = uploaded.height;
    redrawCanvas();
    updatePromptJson();
  };

  image.src = url;
}

function canvasToImageCoords(canvasX, canvasY) {
  if (!state.imageMeta) {
    return { x: 0, y: 0 };
  }

  const rect = canvas.getBoundingClientRect();
  const x = Math.round((canvasX / rect.width) * state.imageMeta.width);
  const y = Math.round((canvasY / rect.height) * state.imageMeta.height);

  return {
    x: clamp(x, 0, state.imageMeta.width - 1),
    y: clamp(y, 0, state.imageMeta.height - 1),
  };
}

function imageToCanvasCoords(imageX, imageY) {
  if (!state.imageMeta) {
    return { x: 0, y: 0 };
  }

  return {
    x: (imageX / state.imageMeta.width) * canvas.width,
    y: (imageY / state.imageMeta.height) * canvas.height,
  };
}

function redrawCanvas() {
  context.clearRect(0, 0, canvas.width, canvas.height);

  if (!state.image) {
    return;
  }

  context.drawImage(state.image, 0, 0, canvas.width, canvas.height);

  const box = state.draftBox || state.box;
  if (box) {
    drawBox(box, Boolean(state.draftBox));
  }

  state.points.forEach((point, index) => {
    drawPoint(point, state.pointLabels[index]);
  });
}

function drawBox(box, isDraft) {
  const start = imageToCanvasCoords(box[0], box[1]);
  const end = imageToCanvasCoords(box[2], box[3]);
  const x = Math.min(start.x, end.x);
  const y = Math.min(start.y, end.y);
  const width = Math.abs(end.x - start.x);
  const height = Math.abs(end.y - start.y);

  context.save();
  context.lineWidth = Math.max(2, canvas.width / 500);
  context.strokeStyle = isDraft ? "#f59f00" : "#1971c2";
  context.setLineDash(isDraft ? [8, 6] : []);
  context.strokeRect(x, y, width, height);
  context.restore();
}

function drawPoint(point, label) {
  const position = imageToCanvasCoords(point[0], point[1]);
  const radius = Math.max(5, Math.min(canvas.width, canvas.height) / 90);

  context.save();
  context.beginPath();
  context.arc(position.x, position.y, radius, 0, Math.PI * 2);
  context.fillStyle = label === 1 ? "#2f9e44" : "#e03131";
  context.strokeStyle = "#ffffff";
  context.lineWidth = Math.max(2, radius / 3);
  context.fill();
  context.stroke();
  context.restore();
}

function getImageCoordsFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  return canvasToImageCoords(event.clientX - rect.left, event.clientY - rect.top);
}

function setActiveTool(tool) {
  state.activeTool = tool;

  document.querySelectorAll("[data-tool]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.tool === tool);
  });
}

function addPoint(point, label) {
  state.points.push([point.x, point.y]);
  state.pointLabels.push(label);
  redrawCanvas();
  updatePromptJson();
}

function normalizeBox(start, end) {
  return [
    Math.min(start.x, end.x),
    Math.min(start.y, end.y),
    Math.max(start.x, end.x),
    Math.max(start.y, end.y),
  ];
}

function clearPrompts() {
  state.points = [];
  state.pointLabels = [];
  state.box = null;
  state.draftBox = null;
  state.isDrawingBox = false;
  redrawCanvas();
  updatePromptJson();
}

function updatePromptJson() {
  promptJson.textContent = JSON.stringify(
    {
      points: state.points,
      point_labels: state.pointLabels,
      box: state.box,
    },
    null,
    2,
  );
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function handleCanvasMouseDown(event) {
  if (!state.imageMeta) {
    return;
  }

  const point = getImageCoordsFromEvent(event);

  if (event.button === 2) {
    event.preventDefault();
    addPoint(point, 0);
    return;
  }

  if (event.button !== 0) {
    return;
  }

  if (state.activeTool === "positive") {
    addPoint(point, 1);
    return;
  }

  if (state.activeTool === "negative") {
    addPoint(point, 0);
    return;
  }

  state.isDrawingBox = true;
  state.draftBoxStart = point;
  state.draftBox = [point.x, point.y, point.x, point.y];
  redrawCanvas();
}

function handleCanvasMouseMove(event) {
  if (!state.isDrawingBox || !state.draftBoxStart) {
    return;
  }

  const point = getImageCoordsFromEvent(event);
  state.draftBox = normalizeBox(state.draftBoxStart, point);
  redrawCanvas();
}

function handleCanvasMouseUp(event) {
  if (!state.isDrawingBox || event.button !== 0) {
    return;
  }

  const point = getImageCoordsFromEvent(event);
  const box = normalizeBox(state.draftBoxStart, point);
  const hasArea = box[2] > box[0] && box[3] > box[1];

  state.box = hasArea ? box : null;
  state.draftBox = null;
  state.draftBoxStart = null;
  state.isDrawingBox = false;
  redrawCanvas();
  updatePromptJson();
}

document.querySelectorAll("[data-tool]").forEach((button) => {
  button.addEventListener("click", () => setActiveTool(button.dataset.tool));
});
document.getElementById("clear-prompts").addEventListener("click", clearPrompts);
document.getElementById("upload-form").addEventListener("submit", uploadImage);
canvas.addEventListener("mousedown", handleCanvasMouseDown);
window.addEventListener("mousemove", handleCanvasMouseMove);
window.addEventListener("mouseup", handleCanvasMouseUp);
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
window.addEventListener("resize", redrawCanvas);

updatePromptJson();
checkBackend();
