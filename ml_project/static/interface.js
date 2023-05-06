// HTML Elements
const leftVideo = document.querySelector("#left-option > video");
const rightVideo = document.querySelector("#right-option > video");

const totalPairCountText = document.getElementById("total-pair-count");
const ratedPairCountText = document.getElementById("rated-pair-count");

const statusText = document.getElementById("status-text");

// Restart both the left and the right videos
async function restartVideos() {
  if ("currentTime" in leftVideo) {
    leftVideo.currentTime = 0;
    leftVideo.play();
  }

  if ("currentTime" in rightVideo) {
    rightVideo.currentTime = 0;
    rightVideo.play();
  }
}

// Register a preference of the user
/** @param {'left' | 'right' | 'equal' | 'skip'} preference */
async function registerPreference(preference) {
  statusText.innerText = "";

  let response;

  try {
    const parameters = new URLSearchParams({
      leftVideo: new URL(leftVideo.src).pathname.split("/").at(-1),
      rightVideo: new URL(rightVideo.src).pathname.split("/").at(-1),
      preference,
    });

    response = await (
      await fetch("/register-preference?" + parameters, {
        method: "POST",
        headers: { Accept: "application/json" },
      })
    ).json();
  } catch (error) {}

  if (!response) {
    statusText.innerText =
      "Invalid response. Please restart the server and try again.";
    return;
  }

  if (!response.nextLeftVideo || !response.nextRightVideo) {
    statusText.innerText =
      "No more samples to rate. Thank you for providing your preferences!";

    leftVideo.src = "";
    rightVideo.src = "";

    return;
  }

  leftVideo.src = `videos/${response.nextLeftVideo}`;
  rightVideo.src = `videos/${response.nextRightVideo}`;

  totalPairCountText.innerText =
    typeof response.totalPairCount === "number" ? response.totalPairCount : "?";
  ratedPairCountText.innerText =
    typeof response.ratedPairCount === "number" ? response.ratedPairCount : "?";
}

// Add the click listeners to the corresponding buttons
document
  .querySelector("#left-option > button")
  .addEventListener("click", () => registerPreference("left"));
document
  .querySelector("#right-option > button")
  .addEventListener("click", () => registerPreference("right"));

document
  .getElementById("equal-button")
  .addEventListener("click", () => registerPreference("equal"));

document
  .getElementById("skip-button")
  .addEventListener("click", () => registerPreference("skip"));

document
  .getElementById("restart-button")
  .addEventListener("click", restartVideos);
