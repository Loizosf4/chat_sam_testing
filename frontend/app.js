async function checkBackend() {
  const status = document.getElementById("status");

  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    status.textContent = `Backend status: ${data.status}`;
  } catch (error) {
    status.textContent = "Backend status: unavailable";
  }
}

checkBackend();
