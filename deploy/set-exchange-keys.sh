#!/bin/sh
# API-ключи бирж для сканера — ТОЛЬКО НА ЧТЕНИЕ. Нужны, чтобы видеть статусы
# ввода/вывода монет по сетям (причина раздвижки цен, задача B2): у Binance,
# Bybit, OKX и BingX эти данные отдаются только по ключу.
#
#   ssh -t arbitrage sh /opt/arbitrage/deploy/set-exchange-keys.sh
#
# При создании ключа на бирже:
#   * права — только чтение (Read / Read-Only). Торговлю и вывод НЕ включать;
#   * ограничение по IP — адрес сервера 212.8.227.93.
# Значения вводятся здесь скрыто и не попадают ни в историю команд, ни в репозиторий.
set -e
ENV=/opt/arbitrage/.env

printf "Биржа (binance / bybit / okx / bingx / gate / kucoin / bitget): "
read EX
case "$EX" in
  binance|bybit|okx|bingx|gate|kucoin|bitget) ;;
  *) echo "Неизвестная биржа: $EX"; exit 1 ;;
esac
P=$(echo "$EX" | tr a-z A-Z)

printf "API key (ввод скрыт): "; stty -echo; read KEY; stty echo; echo
printf "Secret (ввод скрыт): "; stty -echo; read SECRET; stty echo; echo
PASS=""
case "$EX" in
  okx|kucoin|bitget)
    printf "Passphrase (ввод скрыт): "; stty -echo; read PASS; stty echo; echo ;;
esac
[ -n "$KEY" ] && [ -n "$SECRET" ] || { echo "Пустой ключ — ничего не записано."; exit 1; }

touch "$ENV"; chmod 600 "$ENV"
grep -vE "^${P}_API_(KEY|SECRET|PASSPHRASE)=" "$ENV" > "$ENV.tmp" || true
printf '%s_API_KEY=%s\n%s_API_SECRET=%s\n' "$P" "$KEY" "$P" "$SECRET" >> "$ENV.tmp"
[ -n "$PASS" ] && printf '%s_API_PASSPHRASE=%s\n' "$P" "$PASS" >> "$ENV.tmp"
mv "$ENV.tmp" "$ENV"; chmod 600 "$ENV"
echo "Ключ $EX записан в $ENV."

cd /opt/arbitrage
docker compose -p arbitrage -f docker-compose.leadlag.yml up -d >/dev/null 2>&1
echo "Сервисы перезапущены. Проверка ключа появится вместе с задачей B2."
