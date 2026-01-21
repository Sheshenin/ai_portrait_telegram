# Деплой на Render.com

Пошаговая инструкция по развертыванию AI Portrait Telegram Bot на бесплатном хостинге Render.com.

## Подготовка API ключей

Перед деплоем убедитесь, что у вас есть:

### 1. Telegram Bot Token

1. Откройте Telegram и найдите [@BotFather](https://t.me/BotFather)
2. Отправьте команду `/newbot`
3. Следуйте инструкциям (укажите имя и username бота)
4. **Скопируйте токен** - он понадобится позже

### 2. Google AI API Key

1. Перейдите на [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Войдите с вашим Google аккаунтом
3. Нажмите **"Create API Key"**
4. **Скопируйте ключ** - он понадобится позже

⚠️ **Важно**: Храните эти ключи в безопасности! Никому не показывайте.

---

## Деплой на Render.com

### Шаг 1: Создайте аккаунт на Render

1. Перейдите на [render.com](https://render.com)
2. Нажмите **"Get Started"** или **"Sign Up"**
3. Зарегистрируйтесь через GitHub (рекомендуется) или email
4. Подтвердите email, если требуется

### Шаг 2: Подключите GitHub репозиторий

1. В Render Dashboard нажмите **"New +"** → **"Web Service"**
2. Выберите **"Build and deploy from a Git repository"**
3. Нажмите **"Connect GitHub"** (если еще не подключен)
4. Найдите репозиторий `ai_portrait_telegram` и нажмите **"Connect"**

### Шаг 3: Настройте сервис

Render автоматически определит настройки из `render.yaml`, но проверьте:

**Основные настройки:**
- **Name**: `ai-portrait-telegram-bot` (или любое другое)
- **Region**: `Frankfurt` (EU) или `Oregon` (US) - выберите ближайший
- **Branch**: `main` (или вашу основную ветку)
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python bot.py`

**План:**
- Выберите **Free** (бесплатный tier)

### Шаг 4: Добавьте Environment Variables (переменные окружения)

В разделе **"Environment Variables"** добавьте:

| Key | Value | Описание |
|-----|-------|----------|
| `TELEGRAM_BOT_TOKEN` | `ваш_токен_от_BotFather` | Токен Telegram бота |
| `GOOGLE_API_KEY` | `ваш_Google_AI_API_ключ` | Ключ Google AI |
| `GOOGLE_VISION_MODEL` | `gemini-1.5-flash` | Модель для анализа (необязательно) |
| `GOOGLE_IMAGE_MODEL` | `imagen-3.0-generate-001` | Модель для генерации (необязательно) |

⚠️ **Важно**: Обязательно добавьте первые два (TELEGRAM_BOT_TOKEN и GOOGLE_API_KEY)!

### Шаг 5: Деплой

1. Нажмите **"Create Web Service"**
2. Render начнет процесс деплоя (это займет 3-5 минут)
3. Следите за логами в реальном времени

**Процесс деплоя:**
```
==> Cloning from https://github.com/...
==> Running build command 'pip install -r requirements.txt'...
==> Build successful
==> Starting service with 'python bot.py'...
==> AI Portrait Telegram Bot is running...
```

### Шаг 6: Проверка

1. Когда деплой завершится, статус изменится на **"Live"** (зеленая точка)
2. Откройте Telegram
3. Найдите вашего бота по username
4. Отправьте `/start`
5. Если бот ответил - **все работает!** 🎉

---

## Автоматические обновления

После первого деплоя, Render будет **автоматически** обновлять бота при каждом push в GitHub:

1. Внесите изменения в код локально
2. Сделайте `git commit` и `git push`
3. Render автоматически обнаружит изменения и задеплоит новую версию

---

## Мониторинг и логи

### Просмотр логов

1. Откройте ваш сервис в Render Dashboard
2. Перейдите на вкладку **"Logs"**
3. Здесь вы увидите все логи бота в реальном времени

**Полезные логи:**
```
INFO - User 123456 started the bot
INFO - Analyzing reference image: /tmp/reference_123456_abc.jpg
INFO - Portrait generation completed successfully
```

### Мониторинг

- **Events**: История деплоев и событий
- **Metrics**: Использование CPU и памяти (в платных планах)
- **Shell**: Доступ к терминалу сервиса

---

## Ограничения бесплатного плана Render

✅ **Что включено:**
- 750 часов в месяц (достаточно для 24/7 одного сервиса)
- 512 MB RAM
- 0.1 CPU
- Автодеплой из GitHub
- HTTPS и Custom Domain

⚠️ **Ограничения:**
- Сервис "засыпает" после 15 минут неактивности
- "Просыпается" при первом запросе (cold start ~30 секунд)
- Для Telegram бота это не проблема, т.к. он постоянно получает обновления

💡 **Совет**: Для постоянной работы без "засыпания" можно:
- Перейти на платный план (от $7/месяц)
- Использовать внешний сервис для пинга (не рекомендуется для ботов)
- Использовать альтернативу: Koyeb (не засыпает на бесплатном плане)

---

## Устранение неполадок

### Бот не запускается

**Проблема**: Ошибка при деплое

**Решения:**
1. Проверьте логи в разделе "Logs"
2. Убедитесь, что все environment variables добавлены
3. Проверьте, что `TELEGRAM_BOT_TOKEN` и `GOOGLE_API_KEY` корректны
4. Убедитесь, что `requirements.txt` содержит все зависимости

### Ошибка "TELEGRAM_BOT_TOKEN is not set"

**Решение:**
1. Перейдите в настройки сервиса → **Environment**
2. Добавьте переменную `TELEGRAM_BOT_TOKEN` с вашим токеном
3. Нажмите **"Save Changes"**
4. Render автоматически перезапустит сервис

### Ошибка "GOOGLE_API_KEY is not set"

**Решение:**
1. Убедитесь, что вы создали API ключ на [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Добавьте переменную `GOOGLE_API_KEY` в Environment Variables
3. Сохраните и дождитесь перезапуска

### Бот не отвечает на сообщения

**Проблема**: Бот онлайн, но не реагирует

**Решения:**
1. Проверьте логи - есть ли ошибки?
2. Убедитесь, что токен бота правильный
3. Проверьте, что бот не забанен (@BotFather → /mybots → ваш бот)
4. Попробуйте отправить `/start`

### Превышена квота Google API

**Проблема**: Ошибка "Quota exceeded"

**Решения:**
1. Проверьте лимиты на [Google Cloud Console](https://console.cloud.google.com)
2. Подождите до сброса квоты (обычно раз в день)
3. Рассмотрите возможность включения биллинга для увеличения квот

---

## Альтернативные платформы

Если Render.com не подходит, рассмотрите:

### Koyeb
- ✅ Не засыпает на бесплатном плане
- ✅ 512MB RAM бесплатно
- 🔗 [koyeb.com](https://koyeb.com)

### Railway.app
- ⚠️ Требует привязку карты
- ✅ $5 бесплатных кредитов в месяц
- 🔗 [railway.app](https://railway.app)

### Fly.io
- ⚠️ Требует карту для верификации
- ✅ 3 shared VMs бесплатно
- 🔗 [fly.io](https://fly.io)

---

## Полезные команды

### Локальное тестирование перед деплоем

```bash
# Активировать виртуальное окружение
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Установить зависимости
pip install -r requirements.txt

# Создать .env файл
cp .env.example .env
# Отредактировать .env и добавить ключи

# Запустить бота локально
python bot.py
```

### Push изменений в GitHub

```bash
git add .
git commit -m "Your commit message"
git push origin main
```

После push Render автоматически задеплоит изменения.

---

## Безопасность

⚠️ **Важные правила:**

1. **Никогда** не коммитьте `.env` файл в Git
2. **Всегда** используйте Environment Variables в Render для секретов
3. **Не показывайте** API ключи в коде или логах
4. **Регулярно** обновляйте зависимости: `pip install --upgrade -r requirements.txt`

---

## Поддержка

Если возникли проблемы:

1. **Проверьте логи** в Render Dashboard
2. **Изучите документацию** Render: [render.com/docs](https://render.com/docs)
3. **Создайте Issue** в GitHub репозитории
4. **Напишите в поддержку** Render (если проблема на их стороне)

---

**Готово!** Ваш AI Portrait Telegram Bot теперь работает 24/7 в облаке! 🚀
