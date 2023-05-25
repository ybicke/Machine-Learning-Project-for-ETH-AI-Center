// HTML Elements
const statusText = document.getElementById("status-text");

const totalPairCountText = document.getElementById("total-pair-count");
const ratedPairCountText = document.getElementById("rated-pair-count");

const leftVideo = document.getElementById("left-video");
const rightVideo = document.getElementById("right-video");

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

  if (typeof response.ratedPairCount === "number") {
    ratedPairCountText.innerText = response.ratedPairCount;
  }

  if (typeof response.totalPairCount === "number") {
    totalPairCountText.innerText = response.totalPairCount;
  }

  leftVideo.src = response.nextLeftVideo
    ? `videos/${response.nextLeftVideo}`
    : "";
  rightVideo.src = response.nextRightVideo
    ? `videos/${response.nextRightVideo}`
    : "";

  statusText.innerHTML = response.status;
}

// Add the click listeners to the corresponding buttons
document
  .getElementById("left-button")
  .addEventListener("click", () => registerPreference("left"));
document
  .getElementById("right-button")
  .addEventListener("click", () => registerPreference("right"));

document
  .getElementById("equal-button")
  .addEventListener("click", () => registerPreference("equal"));

document
  .getElementById("restart-button")
  .addEventListener("click", restartVideos);

document
  .getElementById("skip-button")
  .addEventListener("click", () => registerPreference("skip"));
