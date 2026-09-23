# -*- coding: utf-8 -*-

"""
Telegram-бот: определяет цвет и тип одежды по фото.
С эко-блоком: след производства + советы по устойчивой моде.

Установка:
    pip install aiogram torch transformers pillow numpy
"""

import asyncio
import io
import logging
import random
from collections import Counter
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)

from config import TOKEN


# ============================================================
# НАСТРОЙКИ
# ============================================================

MODEL_NAME = "patrickjohncyh/fashion-clip"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_FILE_SIZE = 20 * 1024 * 1024
TYPE_CONFIDENCE_THRESHOLD = 0.45
BG_DISTANCE_THRESHOLD = 55.0


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("ClothingBot")


# ============================================================
# СПРАВОЧНИКИ
# ============================================================

COLOR_RU = {
    # Ахроматика
    "black": "чёрный",
    "white": "белый",
    "gray": "серый",
    # Тёплые нейтральные
    "beige": "бежевый",
    "cream": "кремовый",
    "brown": "коричневый",
    "khaki": "хаки",
    # Красные
    "red": "красный",
    "burgundy": "бордовый",
    "pink": "розовый",
    "coral": "коралловый",
    # Тёплые
    "orange": "оранжевый",
    "yellow": "жёлтый",
    # Зелёные
    "olive": "оливковый",
    "green": "зелёный",
    "mint": "мятный",
    # Сине-зелёные и синие
    "turquoise": "бирюзовый",
    "blue": "синий",
    "navy": "тёмно-синий",
    "sky": "голубой",
    # Фиолетовые
    "purple": "фиолетовый",
    "lavender": "лавандовый",
    # Фолбэк
    "unknown": "неизвестно",
}

TYPE_RU = {
    "top": "верх (футболка, рубашка, свитер…)",
    "bottom": "низ (штаны, джинсы, юбка…)",
    "dress": "платье",
    "unknown": "не удалось определить",
}


# ============================================================
# ЭКО-КОНТЕНТ
# ============================================================

# Оценка «эко-следа» производства по типу вещи.
ECO_FOOTPRINT = {
    "top": ("~2700 л воды", "~6 кг CO₂"),
    "bottom": ("~7500 л воды", "~25 кг CO₂"),
    "dress": ("~5000 л воды", "~15 кг CO₂"),
}

ECO_TIPS = [
    "Стирай при 30 °C — экономит до 60 % электроэнергии по сравнению с 60 °C.",
    "Суши вещи на воздухе, а не в машине — минус ~3 кг CO₂ за цикл.",
    "Секонд-хенд: на одну вещь уходит в 10-20 раз меньше ресурсов, чем на новую.",
    "Носи вещь на 9 месяцев дольше — снижение выбросов CO₂ на 20-30 %.",
    "Маленькую дырку можно зашить — это эко-выбор, а не «экономия».",
    "Синтетику стирай в специальном мешке — меньше микропластика в воде.",
    "Сдай ненужное в благотворительность — вещь получит вторую жизнь.",
    "Не покупай «на один раз» — 1/3 одежды в мире уходит на свалку.",
    "Хлопок — самая «водная» ткань. Один килограмм = 10 000 л воды.",
    "Ремонт обуви, подгонка по фигуре — дешевле и экологичнее новой покупки.",
    "Стирай реже: джинсы и свитеры не нужно стирать после каждой носки.",
    "Покупай у локальных брендов — меньше CO₂ на транспортировку.",
]

ECO_FACTS = [
    "👕 Человек в среднем выбрасывает ~30 кг одежды в год.",
    "🌊 На одну пару джинсов уходит до 7500 л воды — это 5 лет питья.",
    "♻️ В мире перерабатывается только ~12 % текстиля.",
    "🏭 Модная индустрия = ~10 % всех выбросов CO₂ на планете.",
    "⏳ Средний срок жизни вещи — 3 года. Продлевай его!",
    "🐟 Микропластик от синтетики находят в рыбе и водопроводной воде.",
    "🛍️ Производится ~100 млрд вещей в год. Треть — на выброс.",
    "🌳 Одна футболка = ~2700 л воды и ~6 кг CO₂ — как 30 км на авто.",
    "🧵 Натуральные ткани разлагаются годами, синтетика — веками.",
    "💧 700 л воды нужно, чтобы вырастить 1 кг хлопка-сырца.",
]


# ============================================================
# МОДЕЛЬ
# ============================================================

clip_model: Optional[CLIPModel] = None
clip_processor: Optional[CLIPProcessor] = None
model_ready = False


def load_clip() -> bool:
    global clip_model, clip_processor, model_ready
    if model_ready:
        return True
    try:
        log.info("Загружаю FashionCLIP на %s…", DEVICE)
        clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        clip_model = CLIPModel.from_pretrained(MODEL_NAME)
        clip_model.to(DEVICE)
        clip_model.eval()
        model_ready = True
        log.info("Модель готова")
        return True
    except Exception:
        log.exception("CLIP не загрузился — будет только цвет")
        return False


# ============================================================
# ЦВЕТ
# ============================================================

def pixel_to_color(r: int, g: int, b: int) -> str:
    """Один RGB-пиксель → имя цвета."""
    mx = max(r, g, b)
    mn = min(r, g, b)
    v = mx / 255.0
    d = mx - mn
    s = 0.0 if mx == 0 else d / mx

    if d == 0:
        h = 0.0
    elif mx == r:
        h = 60 * (((g - b) / d) % 6)
    elif mx == g:
        h = 60 * ((b - r) / d + 2)
    else:
        h = 60 * ((r - g) / d + 4)

    # --- Ахроматика ---
    if v < 0.18:
        return "black"
    if s < 0.08:
        return "white" if v > 0.85 else "gray"

    # --- Кремовый (тёплый почти-белый) ---
    if s < 0.18 and v > 0.85 and 20 <= h <= 70:
        return "cream"

    if s < 0.10:
        return "white" if v > 0.85 else "gray"

    # --- Красный / розовый / бордовый ---
    if h < 15 or h >= 330:
        if v < 0.55:
            return "burgundy"
        if s < 0.70:
            return "pink"
        return "red"

    # --- Красно-оранжевый → коралловый ---
    if h < 25:
        if v < 0.55:
            return "burgundy"
        if s < 0.75 and v > 0.75:
            return "coral"
        return "red"

    # --- Оранжевый / бежевый / коричневый ---
    if h < 45:
        if v < 0.50:
            return "brown"
        if s < 0.40 and v > 0.72:
            return "beige"
        return "orange"

    # --- Жёлтый / хаки / оливковый ---
    if h < 70:
        if v < 0.55:
            return "olive"
        if s < 0.50 and v < 0.75:
            return "khaki"
        return "yellow"

    # --- Зелёный / мятный / хаки / оливковый ---
    if h < 100:
        if v < 0.45:
            return "olive"
        if s < 0.30 and v > 0.80:
            return "mint"
        if s < 0.50 and v < 0.70:
            return "khaki"
        return "green"

    if h < 160:
        if s < 0.30 and v > 0.80:
            return "mint"
        return "green"

    # --- Бирюзовый ---
    if h < 190:
        return "turquoise"

    # --- Голубой / синий / тёмно-синий ---
    if h < 255:
        if v < 0.45 and s > 0.55:
            return "navy"
        if s < 0.40 and v > 0.75:
            return "sky"
        return "blue"

    # --- Фиолетовый / лавандовый ---
    if s < 0.40 and v > 0.78:
        return "lavender"
    return "purple"


def detect_color(image: Image.Image) -> tuple[str, float]:
    """Возвращает (имя_цвета, уверенность 0..1)."""
    img = image.copy()
    img.thumbnail((200, 200))
    w, h = img.size
    if w < 20 or h < 20:
        return "unknown", 0.0

    # Фон — медиана по 4 углам
    cs = max(6, min(w, h) // 10)
    corners = [
        img.crop((0, 0, cs, cs)),
        img.crop((w - cs, 0, w, cs)),
        img.crop((0, h - cs, cs, h)),
        img.crop((w - cs, h - cs, w, h)),
    ]
    bg_pixels = []
    for c in corners:
        bg_pixels.extend(c.getdata())
    bg = np.median(np.array(bg_pixels, dtype=np.float32), axis=0)

    # Центральная зона — там обычно вещь
    center = img.crop(
        (int(w * 0.15), int(h * 0.10), int(w * 0.85), int(h * 0.90))
    )
    arr = np.array(center, dtype=np.float32).reshape(-1, 3)

    dist = np.sqrt(((arr - bg) ** 2).sum(axis=1))
    fg = arr[dist > BG_DISTANCE_THRESHOLD]

    if len(fg) < max(50, len(arr) * 0.05):
        fg = arr

    counter: Counter = Counter()
    for px in fg.astype(np.uint8):
        name = pixel_to_color(int(px[0]), int(px[1]), int(px[2]))
        if name != "unknown":
            counter[name] += 1

    if not counter:
        return "unknown", 0.0

    total = sum(counter.values())

    # Хроматика приоритетнее ахроматики, если её хотя бы 25 %
    achromatic = {"white", "gray", "black"}
    chromatic = {k: v for k, v in counter.items() if k not in achromatic}
    if chromatic:
        best = max(chromatic, key=chromatic.get)
        frac = chromatic[best] / total
        if frac >= 0.25:
            return best, frac

    best, count = counter.most_common(1)[0]
    return best, count / total


# ============================================================
# ТИП
# ============================================================

TYPE_PROMPTS = {
    "top": [
        "a photo of a t-shirt",
        "a photo of a shirt",
        "a photo of a sweater",
        "a photo of a hoodie",
        "a photo of a jacket",
        "a photo of a blouse",
        "a photo of a tank top",
        "a photo of a polo shirt",
    ],
    "bottom": [
        "a photo of pants",
        "a photo of jeans",
        "a photo of trousers",
        "a photo of shorts",
        "a photo of a skirt",
        "a photo of leggings",
    ],
    "dress": [
        "a photo of a dress",
        "a photo of a gown",
        "a photo of a long dress",
        "a photo of a summer dress",
    ],
}


def detect_type(image: Image.Image) -> tuple[str, float]:
    if not model_ready:
        return "unknown", 0.0

    type_names = list(TYPE_PROMPTS.keys())
    flat_prompts: list[str] = []
    owners: list[int] = []
    for ti, tname in enumerate(type_names):
        for p in TYPE_PROMPTS[tname]:
            flat_prompts.append(p)
            owners.append(ti)
    owners_arr = np.array(owners)

    inputs = clip_processor(
        text=flat_prompts,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if hasattr(v, "to")}

    with torch.inference_mode():
        logits = clip_model(**inputs).logits_per_image[0].float().cpu().numpy()

    class_scores = np.array(
        [logits[owners_arr == ti].mean() for ti in range(len(type_names))]
    )
    exp = np.exp(class_scores - class_scores.max())
    probs = exp / exp.sum()

    best = int(np.argmax(probs))
    return type_names[best], float(probs[best])


# ============================================================
# АНАЛИЗ
# ============================================================

def prepare_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise ValueError("Пустой файл")
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            im.verify()
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("Не изображение") from e

    with Image.open(io.BytesIO(image_bytes)) as original:
        im = ImageOps.exif_transpose(original).convert("RGB")
        im.thumbnail((1000, 1000))
        return im.copy()


async def analyze(image_bytes: bytes) -> dict:
    image = await asyncio.to_thread(prepare_image, image_bytes)
    color, color_conf = await asyncio.to_thread(detect_color, image)

    clothing_type, type_conf = "unknown", 0.0
    if model_ready:
        try:
            clothing_type, type_conf = await asyncio.to_thread(detect_type, image)
            if type_conf < TYPE_CONFIDENCE_THRESHOLD:
                clothing_type = "unknown"
        except Exception:
            log.exception("Ошибка CLIP")

    return {
        "color": color,
        "color_conf": color_conf,
        "type": clothing_type,
        "type_conf": type_conf,
    }


# ============================================================
# КЛАВИАТУРЫ
# ============================================================

def kb_start() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📸 Распознать одежду", callback_data="recognize")],
            [InlineKeyboardButton(text="🌱 Эко-совет", callback_data="eco_tip")],
        ]
    )


def kb_result() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📸 Другое фото", callback_data="recognize")],
            [InlineKeyboardButton(text="🌱 Ещё эко-совет", callback_data="eco_tip")],
            [InlineKeyboardButton(text="🏠 В начало", callback_data="home")],
        ]
    )


# ============================================================
# TELEGRAM-ФАЙЛЫ
# ============================================================

async def get_file_id(message: Message) -> Optional[str]:
    if message.photo:
        return message.photo[-1].file_id
    if message.document:
        doc = message.document
        mime = (doc.mime_type or "").lower()
        name = (doc.file_name or "").lower()
        if mime.startswith("image/") or name.endswith(
            (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif")
        ):
            return doc.file_id
    return None


async def download(message: Message, file_id: str) -> bytes:
    tf = await message.bot.get_file(file_id)
    if tf.file_size and tf.file_size > MAX_FILE_SIZE:
        raise ValueError("Файл больше 20 МБ")
    if not tf.file_path:
        raise ValueError("Нет пути к файлу")
    buf = io.BytesIO()
    await message.bot.download_file(tf.file_path, destination=buf)
    data = buf.getvalue()
    if not data:
        raise ValueError("Пусто")
    return data


# ============================================================
# ТЕКСТЫ
# ============================================================

START_TEXT = (
    "👕 <b>Распознавание одежды + эко-советы</b>\n\n"
    "Отправь фото вещи — я определю её <b>цвет</b> и, "
    "если смогу, <b>тип</b> (верх / низ / платье).\n\n"
    "А ещё покажу <b>эко-след</b> производства и дам "
    "совет, как продлить жизнь вещи. 🌍\n\n"
    "Лучше всего работает фото на однотонном фоне, "
    "где вещь целиком в кадре."
)


def format_result(r: dict) -> str:
    color_text = COLOR_RU.get(r["color"], "неизвестно")
    type_text = TYPE_RU.get(r["type"], "неизвестно")

    lines = [
        "✅ <b>Результат</b>\n",
        f"🎨 Цвет: <b>{color_text}</b>",
    ]

    if r["type"] == "unknown":
        lines.append("👕 Тип: <i>не удалось определить</i>")
    else:
        pct = int(r["type_conf"] * 100)
        lines.append(f"👕 Тип: <b>{type_text}</b> ({pct}%)")

    # Эко-след
    footprint = ECO_FOOTPRINT.get(r["type"])
    lines.append("")
    if footprint:
        water, co2 = footprint
        lines.append(
            f"🌍 На производство такой вещи ушло примерно "
            f"<b>{water}</b> и <b>{co2}</b>."
        )
    else:
        lines.append(
            "🌍 Производство одежды — один из самых "
            "ресурсоёмких процессов в мире."
        )

    # Случайный совет
    lines.append(f"💡 <i>{random.choice(ECO_TIPS)}</i>")

    return "\n".join(lines)


def format_eco_tip() -> str:
    tip = random.choice(ECO_TIPS)
    fact = random.choice(ECO_FACTS)
    return (
        "🌱 <b>Эко-совет</b>\n\n"
        f"{tip}\n\n"
        f"<b>А знаешь?</b>\n{fact}"
    )


# ============================================================
# ОБРАБОТКА ФОТО
# ============================================================

async def process_photo(message: Message) -> None:
    status = await message.answer("🔍 Смотрю фото…")

    try:
        file_id = await get_file_id(message)
        if not file_id:
            await status.edit_text(
                "❌ Не вижу изображение. Отправь фото "
                "или файл JPG/PNG/WEBP."
            )
            return

        data = await download(message, file_id)

        if model_ready:
            await status.edit_text("🧠 Анализирую…")
        else:
            await status.edit_text(
                "🧠 Анализирую цвет…\n"
                "<i>(модель типов ещё грузится в фоне)</i>"
            )

        result = await analyze(data)
        await status.edit_text(
            format_result(result),
            reply_markup=kb_result(),
        )

    except ValueError as e:
        log.warning("Ошибка изображения: %s", e)
        await status.edit_text(
            "❌ Не получилось прочитать файл. "
            "Попробуй обычное фото JPG/PNG."
        )
    except Exception:
        log.exception("Ошибка обработки фото")
        await status.edit_text(
            "😔 Ошибка при анализе. Попробуй другое фото."
        )


# ============================================================
# HANDLERS
# ============================================================

async def on_start(message: Message) -> None:
    await message.answer(START_TEXT, reply_markup=kb_start())


async def on_recognize(callback: CallbackQuery) -> None:
    await callback.answer()
    try:
        await callback.message.answer(
            "📸 Отправь фото одежды.\n"
            "Лучше — однотонный фон и вещь целиком в кадре."
        )
    except Exception:
        log.exception("Callback recognize")


async def on_home(callback: CallbackQuery) -> None:
    await callback.answer()
    try:
        await callback.message.answer(START_TEXT, reply_markup=kb_start())
    except Exception:
        log.exception("Callback home")


async def on_eco_tip(callback: CallbackQuery) -> None:
    await callback.answer()
    try:
        await callback.message.answer(
            format_eco_tip(),
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(
                        text="🌱 Ещё совет",
                        callback_data="eco_tip",
                    )],
                    [InlineKeyboardButton(
                        text="📸 Распознать одежду",
                        callback_data="recognize",
                    )],
                ]
            ),
        )
    except Exception:
        log.exception("Callback eco_tip")


async def on_photo(message: Message) -> None:
    await process_photo(message)


async def on_text(message: Message) -> None:
    await message.answer(
        "Отправь фото одежды — определю цвет, тип "
        "и покажу эко-след. 🌍",
        reply_markup=kb_start(),
    )


# ============================================================
# ЗАПУСК
# ============================================================

async def main() -> None:
    bot = Bot(
        token=TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()

    dp.message.register(on_start, CommandStart())
    dp.callback_query.register(on_recognize, F.data == "recognize")
    dp.callback_query.register(on_home, F.data == "home")
    dp.callback_query.register(on_eco_tip, F.data == "eco_tip")
    dp.message.register(on_photo, F.photo)
    dp.message.register(on_photo, F.document)
    dp.message.register(on_text)

    await bot.delete_webhook(drop_pending_updates=True)
    me = await bot.get_me()
    log.info("Запущен как @%s", me.username)

    asyncio.create_task(asyncio.to_thread(load_clip))

    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Остановлено")
