(function () {
  var toggle = document.querySelector("[data-nav-toggle]");
  var nav = document.querySelector("[data-site-nav]");
  if (!toggle || !nav) return;

  function setOpen(open) {
    nav.classList.toggle("is-open", open);
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
  }

  toggle.addEventListener("click", function () {
    setOpen(!nav.classList.contains("is-open"));
  });

  nav.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      setOpen(false);
      toggle.focus();
    }
  });
})();
