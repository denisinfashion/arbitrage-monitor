#!/bin/sh
# Подключение Telegram-бота к сканеру. Значения вводятся здесь, на сервере,
# и не попадают ни в историю команд, ни в репозиторий.
#
#   ssh -t arbitrage sh /opt/arbitrage/deploy/set-telegram.sh
#
# Токен — у @BotFather: /mybots → бот → API Token. ID чата — например,
# у @userinfobot (личный) или у бота в группе.
set -e
ENV=/opt/arbitrage/.env

printf "Токен бота (ввод скрыт): "
stty -echo; read TOKEN; stty echo; echo
printf "ID чата: "
read CHAT

NAME=$(curl -s -m 15 "https://api.telegram.org/bot$TOKEN/getMe" | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print(d['result']['username'] if d.get('ok') else '')")
if [ -z "$NAME" ]; then
  echo "Telegram не принял токен — проверьте и запустите снова."; exit 1
fi
echo "Бот: @$NAME"

OK=$(curl -s -m 15 -d chat_id="$CHAT" --data-urlencode text="✅ Сканер связок подключён к этому чату." \
  "https://api.telegram.org/bot$TOKEN/sendMessage" | python3 -c \
  "import json,sys; print(json.load(sys.stdin).get('ok'))")
if [ "$OK" != "True" ]; then
  echo "Сообщение в чат $CHAT не ушло: напишите боту /start (или добавьте его в группу) и повторите."; exit 1
fi

touch "$ENV"; chmod 600 "$ENV"
grep -vE '^(TELEGRAM_BOT_TOKEN|TELEGRAM_CHAT_ID)=' "$ENV" > "$ENV.tmp" || true
printf 'TELEGRAM_BOT_TOKEN=%s\nTELEGRAM_CHAT_ID=%s\n' "$TOKEN" "$CHAT" >> "$ENV.tmp"
mv "$ENV.tmp" "$ENV"; chmod 600 "$ENV"
echo "Записано в $ENV. Проверочное сообщение отправлено — проверьте чат."

cd /opt/arbitrage
docker compose -p arbitrage -f docker-compose.leadlag.yml up -d >/dev/null 2>&1
echo "Сервисы перезапущены с новыми настройками."
