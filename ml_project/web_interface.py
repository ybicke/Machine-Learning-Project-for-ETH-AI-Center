"""Module for the web interface to provide user preferences."""
from functools import wraps
from glob import glob
from io import TextIOWrapper
from os import path
from pathlib import Path
from random import randrange
from typing import Any, Callable
from uuid import uuid4

from flask import Flask, abort, render_template, request, session

# pylint: disable=fixme
# TODO: move videos closer together

current_path = Path(__file__).parent.resolve()
static_path = path.join(current_path, "static")
videos_path = path.join(static_path, "videos")

output_path = path.join(current_path, "output")
preferences_path = path.join(output_path, "preferences.csv")

# Create output directory and preferences file if they do not exist yet
Path(output_path).mkdir(exist_ok=True)
if not path.exists(preferences_path):
    with open(preferences_path, "w", encoding="utf-8") as output_file:
        output_file.write("User ID;Left Video;Right Video;Preference\n")

# Helper functions


# pylint: disable=too-many-branches
def get_next_videos(preferences_file: TextIOWrapper):
    """Find the next videos to be compared by the user."""
    videos = [path.basename(video) for video in glob(f"{videos_path}/*.mp4")]

    if len(videos) == 0:
        return {}

    videos_set = set(videos)
    user_id = session["userId"]

    # Skip header row
    next(preferences_file)

    # Filter preferences (e.g., videos that don't exist anymore,
    # so we can use the length of list to see if all videos were compared to this one)
    preferences = [
        preference
        for preference in (line.split(";") for line in preferences_file)
        if preference[0] == user_id
        and preference[1] in videos_set
        or preference[2] in videos_set
        or preference[0] != preference[1]
    ]

    preference_pairs = {}

    # Add already rated videos to a dictionary of sets for quick lookups
    for preference in preferences:
        if preference[1] not in preference_pairs:
            preference_pairs[preference[1]] = {preference[2]}
        else:
            preference_pairs[preference[1]].add(preference[2])

        if preference[2] not in preference_pairs:
            preference_pairs[preference[2]] = {preference[1]}
        else:
            preference_pairs[preference[2]].add(preference[1])

    # Find next random video pair (exclusing the already rated ones)
    next_left_video = None
    next_right_video = None

    is_new_pair_found = False

    original_video_count = len(videos)

    while not is_new_pair_found:
        video_count = len(videos)
        video_indices = [randrange(video_count), randrange(video_count)]

        if video_indices[0] == video_indices[1]:
            continue

        next_left_video = videos[video_indices[0]]
        next_right_video = videos[video_indices[1]]

        if (
            next_left_video not in preference_pairs
            or next_right_video not in preference_pairs[next_left_video]
        ):
            is_new_pair_found = True
        else:
            # If all videos were compared to this one,
            # remove it from the video list (so that we don't consider them again)
            if len(preference_pairs[next_left_video]) == original_video_count - 1:
                videos.remove(next_left_video)

            if len(preference_pairs[next_right_video]) == original_video_count - 1:
                videos.remove(next_right_video)

    if next_left_video is None or next_right_video is None:
        return {}

    return {
        "nextLeftVideo": next_left_video,
        "nextRightVideo": next_right_video,
        "ratedPairCount": len(preferences),
        "totalPairCount": int((original_video_count * (original_video_count - 1)) / 2),
    }


# Flask webapp

app = Flask(__name__, template_folder="static", static_url_path="/")
app.secret_key = "MhTOR2a87AKIaMqK0ih0nlG1morh1sZg"


def use_session(route: Callable[[], Any]):
    """Decorate a Flask route to make sure the session has a valid user ID."""

    @wraps(route)
    def decorated_route(*args, **kwargs):
        if (
            "userId" not in session
            or not isinstance(session["userId"], str)
            or session["userId"] == ""
        ):
            session["userId"] = str(uuid4())

        return route(*args, **kwargs)

    return decorated_route


@app.get("/")
@use_session
def home():
    """Load web interface home page."""
    next_videos = {}

    with open(preferences_path, "r", encoding="utf-8") as preferences_file:
        next_videos = get_next_videos(preferences_file)

    user_id = session["userId"]
    error = (
        f'No video was found in "{videos_path}"'
        f'" that was not yet rated by user "{user_id}".'
        if "nextLeftVideo" not in next_videos or "nextRightVideo" not in next_videos
        else ""
    )

    return render_template(
        "interface.html",
        left_video=next_videos["nextLeftVideo"]
        if "nextLeftVideo" in next_videos
        else "",
        right_video=next_videos["nextRightVideo"]
        if "nextRightVideo" in next_videos
        else "",
        rated_pair_count=next_videos["ratedPairCount"],
        total_pair_count=next_videos["totalPairCount"],
        error=error,
    )


@app.post("/register-preference")
@use_session
def register_preference():
    """Add a new entry to the preferences."""
    arguments = request.args
    preference = arguments.get("preference")

    if preference not in ["left", "right", "equal", "skip"]:
        return abort(400, "Invalid preference")

    next_videos = {}

    with open(preferences_path, "a+", encoding="utf-8") as preferences_file:
        if preference in ["left", "right", "equal"]:
            left_video = arguments.get("leftVideo")
            right_video = arguments.get("rightVideo")

            if left_video is None or right_video is None:
                return abort(400, "Invalid videos provided with the preference")

            preferences_file.write(
                f"{session['userId']};{left_video};{right_video};{preference}\n"
            )

        preferences_file.seek(0)
        next_videos = get_next_videos(preferences_file)

    return next_videos
