// Small UX helpers for the prediction form.
document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  const button = document.getElementById("submit-btn");
  if (!form || !button) return;

  form.addEventListener("submit", (event) => {
    // Use the browser's built-in validation messages before sending.
    if (!form.checkValidity()) {
      event.preventDefault();
      form.reportValidity();
      return;
    }
    button.disabled = true;
    button.textContent = "Predicting...";
  });
});
