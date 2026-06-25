let apiBasePromise = null;
const canvas = document.getElementById("image-canvas");
const context = canvas.getContext("2d");
const promptJson = document.getElementById("prompt-json");
const maskCandidates = document.getElementById("mask-candidates");
const finalMasks = document.getElementById("final-masks");
const overlayOpacity = document.getElementById("overlay-opacity");

const state = {
  activeTool: "box",
  image: null,
  imageMeta: null,
  points: [],
  pointLabels: [],
  box: null,
  draftBox: null,
  isDrawingBox: false,
  masks: [],
  selectedMaskIds: new Set(),
  finalMasks: [],
  selectedFinalMaskIds: new Set(),
  activeFinalMaskId: null,
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

async function uploadSelectedImage() {
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
  } finally {
    fileInput.value = "";
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
    state.masks = [];
    state.selectedMaskIds.clear();
    state.finalMasks = [];
    state.selectedFinalMaskIds.clear();
    state.activeFinalMaskId = null;

    canvas.width = uploaded.width;
    canvas.height = uploaded.height;
    redrawCanvas();
    updatePromptJson();
    renderMaskCandidates();
    renderFinalMasks();
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

  drawSelectedMaskOverlays();

  const box = state.draftBox || state.box;
  if (box) {
    drawBox(box, Boolean(state.draftBox));
  }

  state.points.forEach((point, index) => {
    drawPoint(point, state.pointLabels[index]);
  });
}

function drawSelectedMaskOverlays() {
  const opacity = Number(overlayOpacity.value) || 0.45;

  drawMaskGroupOverlays(state.masks, state.selectedMaskIds, opacity);
  drawMaskGroupOverlays(state.finalMasks, state.selectedFinalMaskIds, opacity);
}

function drawMaskGroupOverlays(masks, selectedIds, opacity) {
  masks.forEach((mask) => {
    if (selectedIds.has(mask.mask_id) && mask.overlayCanvas) {
      context.save();
      context.globalAlpha = opacity;
      context.drawImage(mask.overlayCanvas, 0, 0, canvas.width, canvas.height);
      context.restore();
    }
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

async function predictMasks() {
  const status = document.getElementById("status");

  if (!state.imageMeta) {
    status.textContent = "Upload an image before predicting.";
    return;
  }

  if (state.points.length === 0 && !state.box) {
    status.textContent = "Add at least one point or a box before predicting.";
    return;
  }

  status.textContent = "Predicting masks...";

  try {
    const apiBase = await getApiBase();
    const response = await fetch(`${apiBase}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        image_id: state.imageMeta.image_id,
        points: state.points,
        point_labels: state.pointLabels,
        box: state.box,
        multimask_output: true,
      }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    const prediction = await response.json();
    await loadMaskCandidates(prediction.masks);
    status.textContent = `Predicted ${state.masks.length} mask candidate(s).`;
  } catch (error) {
    status.textContent = `Prediction failed: ${error.message}`;
  }
}

async function loadMaskCandidates(masks) {
  state.selectedMaskIds.clear();

  state.masks = await Promise.all(
    masks.map(async (mask) => {
      const loadedMask = await loadMaskImage(mask);
      state.selectedMaskIds.add(mask.mask_id);
      return loadedMask;
    }),
  );

  renderMaskCandidates();
  redrawCanvas();
}

async function mergeSelectedMasks() {
  const status = document.getElementById("status");
  const maskIds = getSelectedOperationMaskIds();

  if (maskIds.length === 0) {
    status.textContent = "Select at least one mask to merge.";
    return;
  }

  status.textContent = "Merging selected masks...";

  try {
    const result = await postJson("/merge_masks", {
      mask_ids: maskIds,
      label: getObjectLabel(),
    });
    await addFinalMask(result);
    state.selectedMaskIds.clear();
    renderMaskCandidates();
    redrawCanvas();
    status.textContent = `Merged mask created: ${result.label || result.mask_id}`;
  } catch (error) {
    status.textContent = `Merge failed: ${error.message}`;
  }
}

async function subtractSelectedMasks() {
  const status = document.getElementById("status");

  if (!state.activeFinalMaskId) {
    status.textContent = "Select a final mask as the active/base mask first.";
    return;
  }

  const subtractMaskIds = getSelectedOperationMaskIds().filter(
    (maskId) => maskId !== state.activeFinalMaskId,
  );

  if (subtractMaskIds.length === 0) {
    status.textContent = "Select at least one other mask to subtract.";
    return;
  }

  status.textContent = "Subtracting selected masks...";

  try {
    const result = await postJson("/subtract_masks", {
      base_mask_id: state.activeFinalMaskId,
      subtract_mask_ids: subtractMaskIds,
      label: getObjectLabel(),
    });
    await addFinalMask(result);
    state.selectedMaskIds.clear();
    renderMaskCandidates();
    redrawCanvas();
    status.textContent = `Cleaned mask created: ${result.label || result.mask_id}`;
  } catch (error) {
    status.textContent = `Subtract failed: ${error.message}`;
  }
}

async function refineActiveMask(operation) {
  const status = document.getElementById("status");

  if (!state.activeFinalMaskId) {
    status.textContent = "Select a final mask as the active/base mask first.";
    return;
  }

  status.textContent = "Refining active mask...";

  try {
    const result = await postJson("/refine_mask", {
      mask_id: state.activeFinalMaskId,
      operation,
      label: getObjectLabel(),
      min_area: Number(document.getElementById("min-area").value) || 100,
      kernel_size: normalizeKernelSize(Number(document.getElementById("kernel-size").value) || 3),
    });
    await addFinalMask(result);
    status.textContent = `Refined mask created: ${result.label || result.mask_id}`;
  } catch (error) {
    status.textContent = `Refine failed: ${error.message}`;
  }
}

async function postJson(path, body) {
  const apiBase = await getApiBase();
  const response = await fetch(`${apiBase}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

async function addFinalMask(mask) {
  const finalMask = await loadMaskImage(mask);
  state.finalMasks.push(finalMask);
  state.selectedFinalMaskIds.clear();
  state.selectedFinalMaskIds.add(finalMask.mask_id);
  state.activeFinalMaskId = finalMask.mask_id;
  renderFinalMasks();
  redrawCanvas();
}

function loadMaskImage(mask) {
  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => {
      resolve({
        ...mask,
        image,
        overlayCanvas: createMaskOverlayCanvas(image),
      });
    };
    image.onerror = () => {
      resolve({ ...mask, image: null, overlayCanvas: null });
    };
    image.src = `data:image/png;base64,${mask.png_base64}`;
  });
}

function getSelectedOperationMaskIds() {
  return [
    ...state.selectedMaskIds,
    ...state.selectedFinalMaskIds,
  ];
}

function getObjectLabel() {
  return document.getElementById("object-label").value.trim() || null;
}

function normalizeKernelSize(value) {
  const rounded = Math.max(3, Math.round(value));
  return rounded % 2 === 0 ? rounded + 1 : rounded;
}

function createMaskOverlayCanvas(maskImage) {
  const overlay = document.createElement("canvas");
  overlay.width = maskImage.naturalWidth;
  overlay.height = maskImage.naturalHeight;

  const overlayContext = overlay.getContext("2d");
  overlayContext.drawImage(maskImage, 0, 0);

  const pixels = overlayContext.getImageData(0, 0, overlay.width, overlay.height);
  for (let index = 0; index < pixels.data.length; index += 4) {
    const alpha = pixels.data[index];
    pixels.data[index] = 25;
    pixels.data[index + 1] = 113;
    pixels.data[index + 2] = 194;
    pixels.data[index + 3] = alpha;
  }
  overlayContext.putImageData(pixels, 0, 0);

  return overlay;
}

function renderMaskCandidates() {
  maskCandidates.innerHTML = "";

  if (state.masks.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No predictions yet.";
    maskCandidates.appendChild(empty);
    return;
  }

  state.masks.forEach((mask, index) => {
    const item = document.createElement("label");
    item.className = "mask-candidate";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.selectedMaskIds.has(mask.mask_id);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        state.selectedMaskIds.add(mask.mask_id);
      } else {
        state.selectedMaskIds.delete(mask.mask_id);
      }
      redrawCanvas();
    });

    const thumbnail = document.createElement("img");
    thumbnail.alt = `Mask candidate ${index + 1}`;
    thumbnail.src = `data:image/png;base64,${mask.png_base64}`;

    const details = document.createElement("div");
    details.className = "mask-candidate-details";
    details.innerHTML = `
      <strong>${mask.mask_id}</strong>
      <span>score: ${mask.score.toFixed(4)}</span>
      <span>area: ${mask.area}</span>
      <span>bbox: ${JSON.stringify(mask.bbox)}</span>
    `;

    item.append(checkbox, thumbnail, details);
    maskCandidates.appendChild(item);
  });
}

function renderFinalMasks() {
  finalMasks.innerHTML = "";

  if (state.finalMasks.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No final masks yet.";
    finalMasks.appendChild(empty);
    return;
  }

  state.finalMasks.forEach((mask, index) => {
    const item = document.createElement("div");
    item.className = "mask-candidate final-mask";

    const controls = document.createElement("div");
    controls.className = "mask-controls";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.title = "Toggle overlay and operation selection";
    checkbox.checked = state.selectedFinalMaskIds.has(mask.mask_id);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        state.selectedFinalMaskIds.add(mask.mask_id);
      } else {
        state.selectedFinalMaskIds.delete(mask.mask_id);
      }
      redrawCanvas();
    });

    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "active-final-mask";
    radio.title = "Use as active/base mask";
    radio.checked = state.activeFinalMaskId === mask.mask_id;
    radio.addEventListener("change", () => {
      state.activeFinalMaskId = mask.mask_id;
      renderFinalMasks();
    });

    controls.append(checkbox, radio);

    const thumbnail = document.createElement("img");
    thumbnail.alt = `Final mask ${index + 1}`;
    thumbnail.src = `data:image/png;base64,${mask.png_base64}`;

    const details = document.createElement("div");
    details.className = "mask-candidate-details";
    details.innerHTML = `
      <strong>${mask.label || mask.mask_id}</strong>
      <span>${mask.mask_id}</span>
      <span>area: ${mask.area}</span>
      <span>bbox: ${JSON.stringify(mask.bbox)}</span>
      <span>${state.activeFinalMaskId === mask.mask_id ? "active/base" : ""}</span>
    `;

    item.append(controls, thumbnail, details);
    finalMasks.appendChild(item);
  });
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
document.getElementById("predict-button").addEventListener("click", predictMasks);
document.getElementById("merge-selected").addEventListener("click", mergeSelectedMasks);
document.getElementById("subtract-selected").addEventListener("click", subtractSelectedMasks);
document.getElementById("fill-holes").addEventListener("click", () => refineActiveMask("fill_holes"));
document.getElementById("remove-small-parts").addEventListener("click", () => refineActiveMask("remove_small_components"));
document.getElementById("smooth-edges").addEventListener("click", () => refineActiveMask("smooth"));
document.getElementById("image-file").addEventListener("change", uploadSelectedImage);
overlayOpacity.addEventListener("input", redrawCanvas);
canvas.addEventListener("mousedown", handleCanvasMouseDown);
window.addEventListener("mousemove", handleCanvasMouseMove);
window.addEventListener("mouseup", handleCanvasMouseUp);
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
window.addEventListener("resize", redrawCanvas);

updatePromptJson();
checkBackend();
