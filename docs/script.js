const galleries = {
  main: {
    root: "assets/main",
    groups: [
      ["00120.png", "00195.png"],
      ["00066.png", "00093.png", "00298.png", "00181.png"],
      ["000027.png", "000028.png", "000030.png", "000056.png"]
    ]
  },
  "corne-featured": {
    root: "assets/corne",
    groups: [
      ["000050.png", "000159.png", "000114.png", "000200.png"]
    ]
  },
  "corne-carousel": {
    root: "assets/corne",
    groups: [
      ["000000.png", "000010.png", "000023.png", "000045.png"],
      ["000075.png", "000141.png", "000152.png", "000197.png"]
    ]
  },
  anime: {
    root: "assets/anime",
    groups: [["00054.png", "00080.png", "00097.png", "00133.png", "00134.png"]]
  },
  text: {
    root: "assets/text",
    groups: [["00053.png", "00058.png", "00067.png", "00076.png", "00079.png"]]
  }
};

function createCard(gallery, filename) {
  const root = gallery.root;
  const card = document.createElement("button");
  card.className = "result-card";
  card.type = "button";
  card.setAttribute("aria-label", `Show transformed result for ${filename}`);

  const before = document.createElement("img");
  before.className = "before";
  before.src = `${root}/before/${filename}`;
  before.alt = `Before ${filename}`;
  before.loading = "lazy";

  const after = document.createElement("img");
  after.className = "after";
  after.src = `${root}/after/${filename}`;
  after.alt = `After ${filename}`;
  after.loading = "lazy";

  card.append(before, after);
  card.addEventListener("click", () => card.classList.toggle("is-active"));
  card.addEventListener("mouseleave", () => card.classList.remove("is-active"));
  return card;
}

function renderGrid(name) {
  const target = document.querySelector(`[data-gallery="${name}"]`);
  const gallery = galleries[name];
  if (!target || !gallery) return;

  gallery.groups.flat().forEach((filename) => {
    target.appendChild(createCard(gallery, filename));
  });
}

function renderCorneCarousel() {
  const target = document.querySelector('[data-gallery="corne-carousel"]');
  const gallery = galleries["corne-carousel"];
  if (!target) return;

  gallery.groups.forEach((group) => {
    const page = document.createElement("div");
    page.className = "carousel-page";
    group.forEach((filename) => page.appendChild(createCard(gallery, filename)));
    target.appendChild(page);
  });

  let pageIndex = 0;
  const total = gallery.groups.length;
  const update = () => {
    target.style.transform = `translateX(-${pageIndex * 100}%)`;
  };
  const next = () => {
    pageIndex = (pageIndex + 1) % total;
    update();
  };
  const prev = () => {
    pageIndex = (pageIndex - 1 + total) % total;
    update();
  };

  document.querySelector("[data-carousel-next]")?.addEventListener("click", next);
  document.querySelector("[data-carousel-prev]")?.addEventListener("click", prev);
  window.setInterval(next, 4200);
}

renderGrid("main");
renderGrid("corne-featured");
renderCorneCarousel();
renderGrid("anime");
renderGrid("text");
