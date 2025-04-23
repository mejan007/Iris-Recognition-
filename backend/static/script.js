function myfunction() {
  const errorMessage = document.getElementById("error-message");

  if (errorMessage) {
    document.addEventListener("keydown", function (event) {
      console.log(event.key);

      if (event.key === "Escape") {
        errorMessage.remove();

        document.removeEventListener("keydown", this);
      }
    });
  }
}
