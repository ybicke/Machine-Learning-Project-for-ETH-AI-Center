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
# TODO: Show error on interface if no videos are found,
# test if reading after append works as expected
# clean up `get_next_videos` function

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


def get_next_videos(preferences_file: TextIOWrapper):
    """Find the next videos to be compared by the user."""
    videos = [path.basename(video) for video in glob(f"{videos_path}/*.mp4")]

    if len(videos) == 0:
        return {}

    videos_set = set(videos)
    # preferences = set(
    #     (line[1], line[2])
    #     for line in (line_string.split(";") for line_string in preferences_file)
    #     if line[0] == session["userId"]
    # )
    preferences = [line.split(";") for line in preferences_file]

    preference_pairs = {}

    # Initialize dictionary with empty sets
    # video_pairs = {preference[1]: set() for preference in preferences}
    # video_pairs = {
    #     **video_pairs,
    #     **{preference[2]: set() for preference in preferences},
    # }

    # Add already rated videos to the dictionary
    for preference in preferences:
        if preference[1] not in videos_set or preference[2] not in videos_set:
            continue

        # Sort key and value, so that each pair is only in the dict once
        keys = (
            [preference[1], preference[2]]
            if preference[1] < preference[2]
            else [preference[2], preference[1]]
        )

        if keys[0] in preference_pairs:
            preference_pairs[keys[0]].add(keys[1])
        else:
            preference_pairs[keys[0]] = set(keys[1])

    next_left_video = None
    next_right_video = None

    is_new_pair_found = False

    while not is_new_pair_found:
        video_count = len(videos)
        video_indices = [randrange(video_count), randrange(video_count)]

        if video_indices[0] == video_indices[1]:
            continue

        next_left_video = videos[video_indices[0]]
        next_right_video = videos[video_indices[1]]

        # Need to sort the key and value again to access the dictionary entry
        sorted_next_videos = (
            [videos[video_indices[0]], videos[video_indices[1]]]
            if video_indices[0] < video_indices[1]
            else [videos[video_indices[1]], videos[video_indices[0]]]
        )

        # if (next_left_video, next_right_video) not in preferences and (
        #     next_right_video,
        #     next_left_video,
        # ) not in preferences:
        #     is_new_pair_found = True
        # elif video_pairs:
        #     pass

        if (
            sorted_next_videos[0] not in preference_pairs
            or preference_pairs[sorted_next_videos[0]] != sorted_next_videos[1]
        ):
            is_new_pair_found = True
        elif len(preference_pairs[sorted_next_videos[0]]) == len(videos):
            videos.remove(sorted_next_videos[0])

    if next_left_video is None or next_right_video is None:
        return {}

    return {"nextLeftVideo": next_left_video, "nextRightVideo": next_right_video}


# Flask webapp

app = Flask(__name__, template_folder="static", static_url_path="/")
app.secret_key = "MhTOR2a87AKIaMqK0ih0nlG1morh1sZg"


def use_session(route: Callable[[], Any]):
    """Decorate a Flask route to make sure the request has a valid session."""

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
        # Skip header row
        next(preferences_file)

        next_videos = get_next_videos(preferences_file)

    return render_template(
        "interface.html",
        left_video=next_videos["nextLeftVideo"]
        if "nextLeftVideo" in next_videos
        else "",
        right_video=next_videos["nextRightVideo"]
        if "nextRightVideo" in next_videos
        else "",
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

        next_videos = get_next_videos(preferences_file)

    return next_videos
