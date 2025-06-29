const fileInput = document.getElementById("fileInput");
const img = document.getElementById("input-image");
const statusDiv = document.getElementById("status");
const resultsDiv = document.getElementById("results");

let net;

async function loadModel() {
  updateStatus("🔄 Loading model...", true);
  try {
    net = await mobilenet.load();
    updateStatus("✅ Model loaded. Upload an image.");
  } catch (err) {
    updateStatus("❌ Failed to load model.");
    console.error(err);
  }
}

function updateStatus(msg, loading = false) {
  statusDiv.innerHTML = loading
    ? `<div class="spinner"></div><p>${msg}</p>`
    : msg;
}

function displayPredictions(predictions) {
  resultsDiv.innerHTML = '';
  predictions.forEach(p => {
    const card = document.createElement("div");
    card.className = "result-card";
    card.innerHTML = `<strong>${p.className}</strong><br>${(p.probability * 100).toFixed(2)}%`;
    resultsDiv.appendChild(card);
  });
}

function handleFile(event) {
  const file = event.target.files[0];
  if (!file) return;
  processImage(file);
}

function handleDrop(event) {
  event.preventDefault();
  const file = event.dataTransfer.files[0];
  if (file) {
    processImage(file);
  }
}

async function processImage(file) {
  img.src = URL.createObjectURL(file);
  img.style.display = 'block';
  updateStatus("🧠 Classifying...", true);

  img.onload = async () => {
    try {
      const predictions = await net.classify(img, 3);
      updateStatus("✅ Classification complete.");
      displayPredictions(predictions);
    } catch (err) {
      updateStatus("❌ Error during classification.");
      console.error(err);
    }
  };
}

loadModel();
