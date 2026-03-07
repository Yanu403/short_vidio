import re

def sanitize_ffmpeg_text(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "")
    text = text.replace('"', "")
    text = text.replace("%", "\\%")
    text = text.replace(",", "")
    text = text.replace("[", "")
    text = text.replace("]", "")
    text = text.replace("=", "")
    text = text.replace("\n", " ")
    return text

def split_hook_lines(text: str) -> str:
    words = text.split()

    if len(words) <= 4:
        return text

    line1 = []
    line2 = []

    for word in words:
        if len(" ".join(line1 + [word])) <= 18:
            line1.append(word)
        else:
            line2.append(word)

    if not line2:
        mid = len(words)//2
        line1 = words[:mid]
        line2 = words[mid:]

    return " ".join(line1) + "\\N" + " ".join(line2)

def optimize_hook(text: str) -> str:
    text = text.upper()
    text = split_hook_lines(text)
    # remove emoji
    text = re.sub(r"[^\w\s,.!?-]", "", text)

    # trim
    text = text.strip()

    if len(text) > 40:
        text = text[:42] + "..."

    return text
