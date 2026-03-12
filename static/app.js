const grid = document.getElementById("grid");
const count = document.getElementById("count");
const search = document.getElementById("search");
const category = document.getElementById("category");
const sort = document.getElementById("sort");
const limit = document.getElementById("limit");
const autorefresh = document.getElementById("autorefresh");
const refreshBtn = document.getElementById("refresh");
const dest = document.getElementById("dest");
const copyBtn = document.getElementById("copy");
const chooseBtn = document.getElementById("choose");

let currentItems = [];
let refreshTimer = null;

async function fetchItems() {
  const params = new URLSearchParams({
    q: search.value || "",
    category: category.value || "",
    sort: sort.value || "latest",
    limit: limit.value || "300"
  });
  const res = await fetch(`/api/items?${params.toString()}`);
  const data = await res.json();
  currentItems = data.items || [];
  render();
  count.textContent = `${data.total || 0} items`;
}

function render() {
  grid.innerHTML = "";
  currentItems.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = "card";

    const img = document.createElement("img");
    img.src = `/api/image?src=${encodeURIComponent(item.src)}`;
    img.loading = "lazy";

    const meta = document.createElement("div");
    meta.className = "meta";
    const row = document.createElement("div");
    row.className = "row";

    const left = document.createElement("div");
    left.textContent = `${item.category || "unclassified"} · ${item.target || "gallery"}`;

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.dataset.idx = idx.toString();

    row.appendChild(left);
    row.appendChild(checkbox);

    const name = document.createElement("div");
    name.className = "name";
    name.textContent = item.src || "";

    meta.appendChild(row);
    meta.appendChild(name);

    card.appendChild(img);
    card.appendChild(meta);

    grid.appendChild(card);
  });
}

async function copySelected() {
  const destPath = dest.value.trim();
  if (!destPath) {
    alert("Enter a destination folder path.");
    return;
  }
  const selected = Array.from(document.querySelectorAll("input[type=checkbox]")
    ).filter(c => c.checked).map(c => currentItems[Number(c.dataset.idx)]);
  if (!selected.length) {
    alert("Select at least one image.");
    return;
  }
  const res = await fetch("/api/copy", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dest: destPath, items: selected })
  });
  const data = await res.json();
  if (data.ok) {
    alert(`Copied ${data.copied} files.`);
  } else {
    alert(`Error: ${data.error || "unknown"}`);
  }
}

refreshBtn.addEventListener("click", fetchItems);
copyBtn.addEventListener("click", copySelected);
chooseBtn.addEventListener("click", async () => {
  const res = await fetch("/api/pick_dir");
  const data = await res.json();
  if (data.ok) {
    dest.value = data.path;
  } else if (data.error) {
    alert(`${data.error}\nIf the picker fails, type the folder path manually.`);
  }
});
autorefresh.addEventListener("change", () => {
  if (refreshTimer) {
    clearInterval(refreshTimer);
    refreshTimer = null;
  }
  const seconds = Number(autorefresh.value || "0");
  if (seconds > 0) {
    refreshTimer = setInterval(fetchItems, seconds * 1000);
  }
});

fetchItems();
