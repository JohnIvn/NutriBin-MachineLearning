const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureBtn = document.getElementById("capture");
const uploadBtn = document.getElementById("upload");
const fileInput = document.getElementById("fileInput");
const jsonEl = document.getElementById("json");
const statsEl = document.getElementById("stats");
const annotatedImg = document.getElementById("annotated");
const enableBtn = document.getElementById("enableCamera");
const disableBtn = document.getElementById("disableCamera");
const reloadBtn = document.getElementById("reloadModel");
const rawDebugEl = document.getElementById("rawdebug");

let currentStream = null;
const modelStatusEl = document.getElementById("modelStatus");

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    currentStream = stream;
    enableBtn.style.display = "none";
    disableBtn.style.display = "";
  } catch (err) {
    alert("Error accessing camera: " + err.message);
  }
}

function stopCamera() {
  if (currentStream) {
    currentStream.getTracks().forEach((t) => t.stop());
    currentStream = null;
    video.srcObject = null;
  }
  enableBtn.style.display = "";
  disableBtn.style.display = "none";
}

enableBtn.addEventListener("click", () => startCamera());
disableBtn.addEventListener("click", () => stopCamera());

captureBtn.addEventListener("click", async () => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(
    async (blob) => {
      const form = new FormData();
      form.append("image", blob, "capture.jpg");
      await sendForm(form);
    },
    "image/jpeg",
    0.9,
  );
});

// camera starts only after user clicks "Enable Camera"

// Fetch server status (model path) and display it
async function refreshStatus() {
  try {
    const res = await fetch("/status");
    if (!res.ok) return;
    const s = await res.json();
    if (s.model_loaded && s.model_path) {
      modelStatusEl.textContent = "Model: " + s.model_path;
    } else if (s.model_path) {
      modelStatusEl.textContent = "Model (not loaded): " + s.model_path;
    } else {
      modelStatusEl.textContent = "Model: none";
    }
  } catch (err) {
    modelStatusEl.textContent = "Model: status unavailable";
  }
}

refreshStatus();

// Reload model button
reloadBtn.addEventListener("click", async () => {
  try {
    reloadBtn.disabled = true;
    const res = await fetch("/reload_model", { method: "POST" });
    const data = await res.json();
    if (res.ok) {
      modelStatusEl.textContent = "Model: " + data.model_path;
    } else {
      modelStatusEl.textContent =
        "Model reload failed: " + (data.error || JSON.stringify(data));
    }
  } catch (err) {
    modelStatusEl.textContent = "Model reload failed";
  } finally {
    reloadBtn.disabled = false;
  }
});
// Upload button handler: send selected file
uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    alert("Select an image file first");
    return;
  }
  const form = new FormData();
  form.append("image", file, file.name);
  await sendForm(form);
});

async function sendForm(form) {
  try {
    const res = await fetch("/predict", { method: "POST", body: form });
    if (!res.ok) {
      const text = await res.text();
      jsonEl.textContent = "Server error: " + text;
      return;
    }
    const data = await res.json();
    // If no predictions, show 'Unidentified'
    if (Array.isArray(data.predictions) && data.predictions.length === 0) {
      jsonEl.textContent = "Unidentified";
      annotatedImg.src = "";
      statsEl.textContent = JSON.stringify(data.stats || {}, null, 2);
      rawDebugEl.textContent = data.raw_debug
        ? JSON.stringify(data.raw_debug, null, 2)
        : "";
    } else {
      jsonEl.textContent = JSON.stringify(data.predictions, null, 2);
      statsEl.textContent = JSON.stringify(data.stats || {}, null, 2);
      rawDebugEl.textContent = data.raw_debug
        ? JSON.stringify(data.raw_debug, null, 2)
        : "";
      if (data.annotated) annotatedImg.src = data.annotated;
    }
  } catch (err) {
    jsonEl.textContent = "Request failed: " + err.message;
  }
}
