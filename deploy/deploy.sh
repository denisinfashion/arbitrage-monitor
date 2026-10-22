#!/bin/sh
# Выкладка арбитражного сканера на общий сервер.
#
#   ssh arbitrage sh /opt/arbitrage/deploy/deploy.sh [ветка]
#
# Всё сканера — с именем arbitrage: каталог /opt/arbitrage, compose-проект
# arbitrage, контейнеры arbitrage-*, журнал /var/log/arbitrage-deploy.log.
# TrueStats (unitrika) живёт в /opt/unitrika и /tmp/deploy-<sha>.sh — их
# этот скрипт не трогает.
set -e
DIR=/opt/arbitrage
LOG=/var/log/arbitrage-deploy.log

# Сборка бывает долгой: работаем в фоне, чтобы обрыв ssh её не прервал.
if [ "$1" != "--bg" ]; then
  nohup sh "$0" --bg "${1:-leadlag-screener}" >>"$LOG" 2>&1 &
  echo "Выкладка сканера идёт в фоне, журнал: $LOG"
  exit 0
fi
BRANCH=$2

echo "== $(date '+%F %T') ветка $BRANCH =="
git -C "$DIR" fetch -q origin "$BRANCH"
git -C "$DIR" checkout -q -B "$BRANCH" "origin/$BRANCH"
git -C "$DIR" log --oneline -1
cd "$DIR"
nice -n 10 docker compose -p arbitrage -f docker-compose.leadlag.yml up -d --build --remove-orphans
docker compose -p arbitrage -f docker-compose.leadlag.yml ps
# образы прежней схемы (по образу на сервис) — теперь один arbitrage-scanner
docker image rm arbitrage-app arbitrage-screener arbitrage-ticks arbitrage-spreads 2>/dev/null || true
rm -f /tmp/deploy-leadlag.sh /tmp/deploy-leadlag.log   # прежний временный скрипт
echo "== OK =="
