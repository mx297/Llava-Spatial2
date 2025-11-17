#!/usr/bin/env python3
import json
import re
import copy
from pathlib import Path
import os

INPUT = Path("/l/users/mohamed.abouelhadid/formatted_questions_frame.json")
OUTPUT = Path("formatted_questions_image.json")

# Find "in frame X of Y" anywhere
FRAME_ANYWHERE_RE = re.compile(
    r"""(?ix)                      # ignore case, verbose
    \b(?:in)\s+frame\s+(\d+)\s+    # 'in frame <num>'
    (?:of)\s+(\d+)                 # 'of <total>'
    \s*,?                          # optional trailing comma/spaces
    """
)

# Remove the exact phrase "These are frames of a video." (case-insensitive), with optional whitespace/newline around it
REMOVE_LEADIN_PHRASE_RE = re.compile(
    r"(?i)These are frames of a video\.\s*\n?", re.UNICODE
)

NEW_INTRO = (
    "You are seeing a single image. \n"
    " In this image,"
)

def transform_samples(samples, img_ext=".jpg", zero_pad=6):
    """
    - Extract frame number from anywhere in the prompt ('in frame X of Y').
    - Remove all occurrences of the phrase 'These are frames of a video.'.
    - Ensure NEW_INTRO is prepended once at the start (if not already present).
    - Remove the first 'in frame X of Y' occurrence to keep the sentence natural.
    - Replace 'video' key with 'image' path built from scene_name and frame index.
    """
    updated = []
    changed = 0
    failures = []

    for idx, item in enumerate(samples):
        obj = copy.deepcopy(item)
        frame_idx = None
        try:
            convs = obj.get("conversations", [])
            human_turn = next((t for t in convs if t.get("from") == "human"), None)
            if not human_turn:
                raise ValueError("No human turn found in conversations")

            text = human_turn.get("value", "")

            # 1) Extract frame index from anywhere in the text
            m = FRAME_ANYWHERE_RE.search(text)
            if not m:
                raise ValueError("Could not parse 'in frame X of Y' anywhere in the prompt")
            frame_idx = int(m.group(1))

            # 2) Remove the phrase "These are frames of a video." wherever it occurs
            text = REMOVE_LEADIN_PHRASE_RE.sub("", text)

            # 3) Ensure our NEW_INTRO is at the very start (avoid duplication)
            #    If it's already there (e.g., from a prior run), don't prepend again.
            if not text.startswith(NEW_INTRO):
                text = f"{NEW_INTRO}\n{text}"

            # 4) Remove the first occurrence of "in frame X of Y," from the remaining text
            text = FRAME_ANYWHERE_RE.sub("", text, count=1)

            # Normalize any double spaces/newlines created by removals
            text = re.sub(r"[ \t]{2,}", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

            human_turn["value"] = text

            # 5) Swap keys and build image path
            scene = obj["scene_name"]
            base = f"/l/users/mohamed.abouelhadid/sampled/color/train/{scene}"
            filename = sorted(os.listdir(base))[frame_idx - 1]
            # filename = (
            #     f"{frame_idx:0{zero_pad}d}{img_ext}" if zero_pad else f"{frame_idx}{img_ext}"
            # )
            image_path = os.path.join(f"sampled/color/train/{scene}",filename) #f"{base}/{filename}"

            obj.pop("video", None)
            obj["image"] = image_path

            changed += 1
            updated.append(obj)

        except Exception as e:
            failures.append(
                {
                    "index": idx,
                    "id": item.get("id"),
                    "scene_name": item.get("scene_name"),
                    "error": str(e),
                    "last_seen_frame_idx": frame_idx,
                }
            )
            updated.append(item)

    return updated, changed, failures

def main():
    if not INPUT.exists():
        raise SystemExit(f"Input file not found: {INPUT.resolve()}")

    data = json.loads(INPUT.read_text())
    if not isinstance(data, list):
        raise SystemExit("Expected top-level JSON array/list.")

    transformed, changed, failures = transform_samples(data)

    OUTPUT.write_text(json.dumps(transformed, ensure_ascii=False, indent=2))
    print(f"Wrote: {OUTPUT}  (modified {changed}/{len(data)} items)")
    if failures:
        print("\nItems that could not be transformed (kept original):")
        for f in failures[:20]:
            print(f"- idx={f['index']} id={f.get('id')} scene={f.get('scene_name')} error={f['error']}")
        if len(failures) > 20:
            print(f"... and {len(failures)-20} more")

if __name__ == "__main__":
    main()
