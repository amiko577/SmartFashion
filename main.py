# main.py
# -*- coding: utf-8 -*-
"""
ReWear AI — Telegram-бот эко-гардероба против фаст-фэшн.
Проект для хакатона VentureHack 2026.

Стек: Python 3.10+, aiogram 3.x
Автор: Lead Developer
"""

import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardRemove,
)

# ============================================================
#  КОНФИГУРАЦИЯ
# ============================================================

# 🔑 ВСТАВЬ СВОЙ ТОКЕН, полученный у @BotFather
from config import TOKEN
# Настройка логирования — чтобы видеть, что происходит
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ReWearAI")


# ============================================================
#  FSM — состояния диалога
# ============================================================

class ReWearStates(StatesGroup):
    """Машина состояний: бот понимает, чего именно ждёт от пользователя."""
    waiting_shop_photo = State()   # ждём фото вещи из магазина
    waiting_closet_photo = State() # ждём фото вещи из шкафа


# ============================================================
#  INLINE-КЛАВИАТУРЫ
# ============================================================

def main_menu_kb() -> InlineKeyboardMarkup:
    """Главное меню бота."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(
                text="🛑 Стоп-покупка (анализ вещи в магазине)",
                callback_data="menu_stop_buy",
            )],
            [InlineKeyboardButton(
                text="✨ Вторая жизнь вещи (стиль из шкафа)",
                callback_data="menu_second_life",
            )],
            [InlineKeyboardButton(
                text="📊 Мой эко-след (статистика)",
                callback_data="menu_stats",
            )],
        ]
    )


def back_to_menu_kb() -> InlineKeyboardMarkup:
    """Кнопка «в меню» после завершения сценария."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⬅️ В главное меню", callback_data="menu_back")],
        ]
    )


def cancel_kb() -> InlineKeyboardMarkup:
    """Кнопка отмены во время ожидания фото."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="❌ Отмена", callback_data="menu_back")],
        ]
    )


# ============================================================
#  ТЕКСТЫ
# ============================================================

START_TEXT = (
    "🌿 <b>Добро пожаловать в ReWear AI</b> 🌿\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "Я — твой персональный <b>ИИ-стилист</b> и <b>эко-защитник</b>.\n\n"
    "Моя миссия проста:\n"
    "👗 помогать покупать <b>меньше</b>,\n"
    "♻️ носить то, что <b>уже есть</b>,\n"
    "🌍 и снижать твой <b>углеродный след</b>.\n\n"
    "Каждый отказ от ненужной покупки = минус ~14 кг CO₂ и плюс деньги в кармане 💸\n\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "Выбери, с чего начнём 👇"
)

SHOP_PROMPT = (
    "🛑 <b>Режим «Стоп-покупка»</b>\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "Пришли 📸 <b>фото вещи</b>, которую хочешь купить в масс-маркете.\n\n"
    "Я проверю состав ткани, посчитаю реальную стоимость носки "
    "и честно скажу: стоит ли она твоих денег 💭"
)

CLOSET_PROMPT = (
    "✨ <b>Режим «Вторая жизнь вещи»</b>\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "Пришли 📸 <b>фото вещи</b> из своего шкафа, которую ты забросил(а).\n\n"
    "Я соберу для тебя <b>3 стильных образа</b> 2026 года "
    "и верну вещи место в гардеробе 💫"
)

STATS_TEXT = (
    "📊 <b>Твой эко-след</b>\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "🌿 Сэкономлено CO₂: <b>42 кг</b>\n"
    "💵 Сохранено денег: <b>45 000 тенге</b>\n"
    "🛍️ Отменено импульсивных покупок: <b>3</b>\n"
    "✨ Возвращено вещей в оборот: <b>7</b>\n\n"
    "🏆 Твой эко-ранг: <b>«Осознанный трендсеттер»</b>\n\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "<i>Ты в топ-15% пользователей по влиянию на планету. "
    "Так держать! 🌍</i>"
)


# ============================================================
#  УТИЛИТЫ
# ============================================================

async def safe_delete(message: Message) -> None:
    """Безопасно удаляет сообщение (не падает, если нет прав)."""
    try:
        await message.delete()
    except Exception:
        pass


# ============================================================
#  ХЕНДЛЕРЫ: /start и навигация
# ============================================================

async def cmd_start(message: Message, state: FSMContext) -> None:
    """/start — приветствие и главное меню."""
    await state.clear()
    await message.answer(START_TEXT, reply_markup=main_menu_kb())


async def cb_back_to_menu(callback: CallbackQuery, state: FSMContext) -> None:
    """Возврат в главное меню (из любого места)."""
    await state.clear()
    await callback.message.edit_text(
        START_TEXT, reply_markup=main_menu_kb()
    )
    await callback.answer()


# ============================================================
#  СЦЕНАРИЙ 1: СТОП-ПОКУПКА
# ============================================================

async def cb_stop_buy(callback: CallbackQuery, state: FSMContext) -> None:
    """Пользователь нажал «Стоп-покупка» — переводим в ожидание фото."""
    await state.set_state(ReWearStates.waiting_shop_photo)
    await callback.message.edit_text(SHOP_PROMPT, reply_markup=cancel_kb())
    await callback.answer()


async def handle_shop_photo(message: Message, state: FSMContext) -> None:
    """Получили фото вещи из магазина — выдаём разбор."""
    # Защита: убеждаемся, что у сообщения есть фото
    if not message.photo:
        await message.answer("🤔 Похоже, это не фото. Пришли, пожалуйста, изображение вещи.")
        return

    # Статусное сообщение — имитация «анализа»
    status = await message.answer(
        "🔍 <i>Анализируем состав ткани и экологический след...</i>\n"
        "⏳ <i>Проверяем твой цифровой гардероб...</i>"
    )
    await asyncio.sleep(1.4)

    await status.edit_text(
        "🧠 <i>Сверяем с трендами 2026 и твоим шкафом...</i>"
    )
    await asyncio.sleep(1.0)

    # Итоговый ответ (в реальном продукте эти данные пришли бы из CV + LLM)
    analysis = (
        "🧵 <b>Анализ состава:</b>\n"
        "Полиэстер <b>90%</b>, эластан <b>10%</b>.\n"
        "<i>Классический фаст-фэшн: быстро теряет форму, "
        "при стирке выделяет микропластик.</i>\n\n"
        "💰 <b>Финансовый расчёт:</b>\n"
        "Реальная стоимость одной носки — <b>~12 000 тг</b> за 1 выход.\n"
        "<i>Тренд уйдёт через месяц — вещь «сгорит» в шкафу.</i>\n\n"
        "🛑 <b>Вердикт ИИ:</b>\n"
        "«Стоп! Не покупай. У тебя в шкафу висит очень похожий "
        "<b>серый кардиган</b>».\n\n"
        "💡 <b>Альтернатива:</b>\n"
        "Скомбинируй старый кардиган с джинсами — получится "
        "трендовый лук 2026 года.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "✅ Ты сэкономил(а) <b>15 000 тенге</b> и <b>14 кг CO₂</b>! 🌍"
    )

    await status.edit_text(analysis, reply_markup=back_to_menu_kb())
    await state.clear()


# ============================================================
#  СЦЕНАРИЙ 2: ВТОРАЯ ЖИЗНЬ ВЕЩИ
# ============================================================

async def cb_second_life(callback: CallbackQuery, state: FSMContext) -> None:
    """Пользователь нажал «Вторая жизнь вещи» — ждём фото из шкафа."""
    await state.set_state(ReWearStates.waiting_closet_photo)
    await callback.message.edit_text(CLOSET_PROMPT, reply_markup=cancel_kb())
    await callback.answer()


async def handle_closet_photo(message: Message, state: FSMContext) -> None:
    """Получили фото вещи из шкафа — генерируем капсулу образов."""
    # Защита от отсутствия фото
    if not message.photo:
        await message.answer("🤔 Похоже, это не фото. Пришли, пожалуйста, изображение вещи.")
        return

    status = await message.answer(
        "🔍 <i>Распознаём вещь и её цветовую палитру...</i>"
    )
    await asyncio.sleep(1.3)

    await status.edit_text(
        "🎨 <i>Подбираем образы под твой стиль...</i>"
    )
    await asyncio.sleep(1.0)

    result = (
        "🧥 <b>Распознано:</b> Чёрный оверсайз-пиджак\n\n"
        "🎨 <b>ИИ-капсула (3 образа 2026):</b>\n\n"
        "🏢 <b>Офис:</b>\n"
        "С классическими брюками-палаццо и белой базовой футболкой.\n\n"
        "☕️ <b>Casual:</b>\n"
        "С винтажными джинсами, худи и кедами.\n\n"
        "🎉 <b>Вечеринка:</b>\n"
        "Поверх шёлкового платья-комбинации с грубыми ботинками.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🌱 <b>Эко-бонус:</b>\n"
        "Ты подарил(а) вещи вторую жизнь. В этот раз планета "
        "спасена от нового производства! 💚"
    )

    await status.edit_text(result, reply_markup=back_to_menu_kb())
    await state.clear()


# ============================================================
#  СЦЕНАРИЙ 3: СТАТИСТИКА
# ============================================================

async def cb_stats(callback: CallbackQuery, state: FSMContext) -> None:
    """Показываем геймифицированную статистику."""
    await state.clear()
    await callback.message.edit_text(STATS_TEXT, reply_markup=back_to_menu_kb())
    await callback.answer()


# ============================================================
#  ОБРАБОТКА «НЕ-ФОТО» В РЕЖИМАХ ОЖИДАНИЯ
# ============================================================

async def non_photo_in_shop(message: Message) -> None:
    """Пользователь в режиме стоп-покупки прислал не фото."""
    await message.answer(
        "📸 Я жду именно <b>фото вещи</b>.\n"
        "Пожалуйста, отправь картинку или нажми «Отмена»."
    )


async def non_photo_in_closet(message: Message) -> None:
    """Пользователь в режиме второй жизни прислал не фото."""
    await message.answer(
        "📸 Я жду именно <b>фото вещи из шкафа</b>.\n"
        "Пожалуйста, отправь картинку или нажми «Отмена»."
    )


# ============================================================
#  ЗАПУСК
# ============================================================

async def main() -> None:
    """Точка входа: инициализация бота, диспетчера, регистрация хендлеров."""
    bot = Bot(
        token=TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()

    # --- Команды ---
    dp.message.register(cmd_start, CommandStart())

    # --- Навигация по меню ---
    dp.callback_query.register(cb_back_to_menu, F.data == "menu_back")
    dp.callback_query.register(cb_stop_buy,     F.data == "menu_stop_buy")
    dp.callback_query.register(cb_second_life,  F.data == "menu_second_life")
    dp.callback_query.register(cb_stats,        F.data == "menu_stats")

    # --- Фото в сценарии «Стоп-покупка» ---
    dp.message.register(
        handle_shop_photo,
        ReWearStates.waiting_shop_photo,
        F.photo,
    )
    dp.message.register(
        non_photo_in_shop,
        ReWearStates.waiting_shop_photo,
    )

    # --- Фото в сценарии «Вторая жизнь» ---
    dp.message.register(
        handle_closet_photo,
        ReWearStates.waiting_closet_photo,
        F.photo,
    )
    dp.message.register(
        non_photo_in_closet,
        ReWearStates.waiting_closet_photo,
    )

    # --- Фолбэк: любой текст вне сценария ---
    @dp.message()
    async def fallback(message: Message) -> None:
        await message.answer(
            "Я понимаю только фото и кнопки меню 🌿\n"
            "Нажми /start, чтобы открыть главное меню.",
            reply_markup=ReplyKeyboardRemove(),
        )

    logger.info("🌿 ReWear AI запущен. Жду сообщений...")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен.")


# ============================================================
#  ИНСТРУКЦИЯ ПО ЗАПУСКУ
# ============================================================
#
# 1. Установи зависимости:
#       pip install aiogram==3.*
#
# 2. Получи токен у @BotFather в Telegram и вставь его в переменную TOKEN.
#
# 3. Запусти скрипт:
#       python main.py
#
# 4. Открой своего бота в Telegram и отправь /start.
#
# ============================================================