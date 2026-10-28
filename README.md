# Crypto Strategy Lab — расширенная версия

Веб-интерфейс на Streamlit для исследования криптостратегий. Не является инвестиционной рекомендацией и не отправляет реальные ордера.

## Возможности
- Публичные OHLCV-котировки: Binance, Bybit, BingX (доступность зависит от региона и API биржи).
- Spot и линейные USDT-фьючерсы для бэктеста.
- 8 стратегий: EMA crossover, RSI mean reversion, Donchian breakout, MACD trend, Bollinger reversion, ATR trend filter, EMA + RSI confirmation, Buy & Hold.
- Настройки EMA, RSI и Donchian; комиссии, проскальзывание, размер позиции, плечо и модельное funding.
- Метрики доходности, CAGR, просадки, Sharpe, Sortino, Profit Factor, сделки и графики.
- Последовательная walk-forward-оценка на нескольких временных отрезках с фиксированными параметрами.
- Paper trading для спота: виртуальный баланс, симуляция покупки/продажи, журнал и защита от повторной обработки одной свечи.
- Экспорт метрик, сделок, свечей, настроек эксперимента и журнала paper trading в CSV/JSON.

## Запуск локально
```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Рекомендуется Python 3.11 или 3.12.

## Тесты
```bash
python -m unittest discover -s tests -v
```

## Важные ограничения
- Биржевые API могут быть заблокированы или менять форматы ответов. После деплоя проверяйте каждый рынок и биржу отдельно.
- Walk-forward использует фиксированные параметры, а не автоматический подбор параметров внутри каждого train-окна. Это последовательная оценка поведения стратегии, но не полноценная оптимизация/валидация производственной системы.
- Paper trading обрабатывает свечу по кнопке и только моделирует long-only spot. Он не работает непрерывно в фоне, не отправляет ордера и не использует API-ключи.
- Paper-журнал хранится в локальном SQLite-файле `paper_trading.db`. На Streamlit Community Cloud локальный файл может исчезнуть при пересоздании контейнера; для долговременного журнала нужен внешний PostgreSQL/managed DB.
- Нет реального автотрейдинга, хранения ключей, контроля минимального лота, точной модели комиссий, ликвидаций, ордербука или фактической истории funding.
- Исторические результаты и paper trading не гарантируют будущую доходность.

## Automatic strategy research (v3)
The app includes an automatic research tab. It evaluates a compact, predeclared candidate set, selects a configuration using only the expanding historical training window, and tests the selection on the next unseen OOS segment. The selection is repeated across sequential folds. It reports OOS return, drawdown, Sharpe, selected rules and a conservative verdict. It may explicitly conclude that no reliable edge was found.

This is not a claim of profitable trading. The candidate universe is limited and OHLCV candles do not include order-book depth, exact fills, historical funding, liquidation mechanics or exchange-specific constraints. Treat results as screening evidence only; require persistent paper trading and out-of-sample confirmation before considering live execution. Automatic research does not place orders.
