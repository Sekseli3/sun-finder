"""Build a small animated diagram of Sunfinder's local planner request flow."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


APP_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = APP_ROOT / "docs" / "request-flow.gif"
WIDTH = 1320
HEIGHT = 770
BACKGROUND = "#172a38"
PAPER = "#fbfaf5"
INK = "#253544"
MUTED = "#aebbc2"
GOLD = "#f4b854"
GOLD_DEEP = "#d98022"
GREEN = "#46a276"
LINE = "#567080"


BOXES = {
    "user": (45, 155, 270, 290),
    "intent": (330, 155, 590, 290),
    "planner": (650, 135, 925, 310),
    "weather": (160, 400, 455, 545),
    "buildings": (530, 400, 825, 545),
    "rag": (900, 400, 1195, 545),
    "rank": (365, 620, 645, 735),
    "writer": (700, 620, 960, 735),
    "result": (1015, 620, 1275, 735),
}


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    )
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


TITLE_FONT = load_font(31, bold=True)
SUBTITLE_FONT = load_font(17)
BOX_TITLE_FONT = load_font(17, bold=True)
BOX_BODY_FONT = load_font(14)
BOX_TINY_FONT = load_font(12)
FOOTER_FONT = load_font(15, bold=True)


def draw_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], lines: list[str], *, title: bool = False) -> None:
    font = BOX_TITLE_FONT if title else BOX_BODY_FONT
    line_height = 22 if title else 19
    total_height = line_height * len(lines)
    y = (box[1] + box[3] - total_height) / 2
    for line in lines:
        left, _, right, _ = draw.textbbox((0, 0), line, font=font)
        draw.text(((box[0] + box[2] - (right - left)) / 2, y), line, fill=INK if title else "#4c5e68", font=font)
        y += line_height


def draw_card(
    draw: ImageDraw.ImageDraw,
    key: str,
    title: list[str],
    body: list[str],
    *,
    active: bool,
    pulse: bool,
) -> None:
    left, top, right, bottom = BOXES[key]
    shadow = 7 if active else 4
    draw.rounded_rectangle((left + shadow, top + shadow, right + shadow, bottom + shadow), radius=18, fill="#0d1c27")
    fill = "#fff8df" if active else PAPER
    outline = GOLD if active else "#c7d0d0"
    width = 4 if active and pulse else 2
    draw.rounded_rectangle((left, top, right, bottom), radius=18, fill=fill, outline=outline, width=width)
    if active:
        draw.rounded_rectangle((left + 12, top + 12, right - 12, top + 18), radius=3, fill=GOLD_DEEP)
    draw_centered(draw, (left + 13, top + 26, right - 13, top + 65), title, title=True)
    draw_centered(draw, (left + 13, top + 67, right - 13, bottom - 12), body)


def arrow(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], *, active: bool) -> None:
    color = GOLD if active else LINE
    width = 5 if active else 3
    draw.line(points, fill=color, width=width, joint="curve")
    tail = points[-2]
    head = points[-1]
    if head[0] == tail[0]:
        direction = 1 if head[1] > tail[1] else -1
        triangle = [(head[0], head[1]), (head[0] - 8, head[1] - 13 * direction), (head[0] + 8, head[1] - 13 * direction)]
    else:
        direction = 1 if head[0] > tail[0] else -1
        triangle = [(head[0], head[1]), (head[0] - 13 * direction, head[1] - 8), (head[0] - 13 * direction, head[1] + 8)]
    draw.polygon(triangle, fill=color)


def draw_frame(active_step: str, pulse: bool) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((44, 36), "Sunfinder local planner request flow", fill=PAPER, font=TITLE_FONT)
    draw.text((44, 77), "Qwen 8B parses and writes. Qwen 0.6B embeds. Python fetches facts and decides the ranking.", fill=MUTED, font=SUBTITLE_FONT)

    progress = {
        "user": 1,
        "intent": 2,
        "facts": 3,
        "rank": 4,
        "writer": 5,
        "result": 6,
    }[active_step]
    arrow(draw, [(270, 222), (330, 222)], active=progress >= 2)
    arrow(draw, [(590, 222), (650, 222)], active=progress >= 3)
    arrow(draw, [(787, 310), (787, 355), (307, 355), (307, 400)], active=progress >= 3)
    arrow(draw, [(787, 310), (787, 400)], active=progress >= 3)
    arrow(draw, [(787, 310), (787, 355), (1047, 355), (1047, 400)], active=progress >= 3)
    arrow(draw, [(307, 545), (307, 580), (505, 580), (505, 620)], active=progress >= 4)
    arrow(draw, [(677, 545), (677, 580), (505, 580), (505, 620)], active=progress >= 4)
    arrow(draw, [(1047, 545), (1047, 580), (505, 580), (505, 620)], active=progress >= 4)
    arrow(draw, [(645, 677), (700, 677)], active=progress >= 5)
    arrow(draw, [(960, 677), (1015, 677)], active=progress >= 6)

    cards = {
        "user": (["1 · User prompt"], ["“Beer near", "Eerikinkatu”"]),
        "intent": (["2 · Qwen 8B"], ["extracts place, time", "and venue type", "as validated JSON"]),
        "planner": (["3 · Python planner"], ["validates the request", "and coordinates", "the evidence"]),
        "weather": (["Weather + nowcast"], ["Open-Meteo inputs", "Bayesian one-hour", "open-point estimate"]),
        "buildings": (["Building geometry"], ["Helsinki WFS", "projected shade", "at now, +30, +60"]),
        "rag": (["Qwen 0.6B RAG"], ["embeds the request", "compares with prebuilt", "venue vectors"]),
        "rank": (["4 · Deterministic rank"], ["building shade + distance", "or distance only", "when geometry fails"]),
        "writer": (["5 · Qwen 8B writes"], ["gets labelled facts", "not live API access", "or ranking internals"]),
        "result": (["6 · Browser response"], ["answer + 3 places", "source links", "clear caveats"]),
    }
    active_keys = {
        "user": {"user"},
        "intent": {"intent"},
        "facts": {"planner", "weather", "buildings", "rag"},
        "rank": {"rank"},
        "writer": {"writer"},
        "result": {"result"},
    }[active_step]
    for key, (title, body) in cards.items():
        draw_card(draw, key, title, body, active=key in active_keys, pulse=pulse)

    footer = "Qwen 8B parses + writes. Qwen 0.6B embeds. Python fetches weather and buildings."
    draw.rounded_rectangle((45, 744, 690, 764), radius=10, fill="#203b4c")
    draw.text((59, 746), footer, fill="#d8e8df", font=FOOTER_FONT)
    return image


def main() -> None:
    frames: list[Image.Image] = []
    for step in ("user", "intent", "facts", "rank", "writer", "result"):
        frames.extend((draw_frame(step, False), draw_frame(step, True)))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=[450, 700] * 6,
        loop=0,
        disposal=2,
        optimize=True,
    )
    print(f"Wrote {OUTPUT_PATH.relative_to(APP_ROOT)}")


if __name__ == "__main__":
    main()
