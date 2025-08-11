import os
import psycopg2
from psycopg2.extras import RealDictCursor
import sqlite3
import hashlib
import random
import json
import time
import logging
import os
import re
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Updater, CommandHandler, CallbackContext, CallbackQueryHandler, MessageHandler, filters,
    JobQueue
)
from cryptography.fernet import Fernet
from PIL import Image, ImageDraw, ImageFont
from bs4 import BeautifulSoup
import snscrape.modules.twitter as sntwitter
from sklearn.ensemble import IsolationForest
from fake_useragent import UserAgent

BOT_TOKEN = os.environ.get("BOT_TOKEN")
DATABASE_URL = os.environ.get("DATABASE_URL")
ADMIN_ID = os.environ.get("ADMIN_ID")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable is required")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

# ===== إعدادات النظام =====
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ===== قاعدة البيانات =====
conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
c = conn.cursor()

# إنشاء الجداول
c.execute )CREATE TABLE IF NOT EXISTS users 
    user_id BIGINT PRIMARY KEY,
    username TEXT,
    join_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    points INTEGER DEFAULT 0,
    level INTEGER DEFAULT 1,
    referral_code TEXT,
    referrer_id BIGINT,
    verified BOOLEAN DEFAULT FALSE,
    total_earned INTEGER DEFAULT 0,
    total_spent INTEGER DEFAULT 0,
    wallet_address TEXT,
    free_recommendations_used INTEGER DEFAULT 0,
    trust_score INTEGER DEFAULT 100,
    escrow_balance REAL DEFAULT 0,
    debt REAL DEFAULT 0
(

c.execute('''CREATE TABLE IF NOT EXISTS referrals (
    referral_id INTEGER PRIMARY KEY AUTOINCREMENT,
    referrer_id INTEGER,
    referred_id INTEGER,
    level INTEGER,
    date TEXT DEFAULT CURRENT_TIMESTAMP,
    verified BOOLEAN DEFAULT FALSE
)''')

c.execute('''CREATE TABLE IF NOT EXISTS transactions (
    tx_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    points INTEGER,
    type TEXT,
    date TEXT DEFAULT CURRENT_TIMESTAMP,
    description TEXT
)''')

c.execute('''CREATE TABLE IF NOT EXISTS quests (
    quest_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    quest_type TEXT,
    progress INTEGER DEFAULT 0,
    completed BOOLEAN DEFAULT FALSE,
    date TEXT DEFAULT CURRENT_TIMESTAMP
)''')

c.execute('''CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    symbol TEXT,
    name TEXT,
    recommendation TEXT,
    reason TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    profit REAL DEFAULT 0,
    verified BOOLEAN DEFAULT FALSE
)''')

c.execute('''CREATE TABLE IF NOT EXISTS audits (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER,
    auditor_id INTEGER,
    vote TEXT,
    comments TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)''')

c.execute('''CREATE TABLE IF NOT EXISTS user_achievements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    achievement_id TEXT,
    progress INTEGER DEFAULT 0
)''')

c.execute('''CREATE TABLE IF NOT EXISTS achievements_unlocked (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    achievement_id TEXT,
    date TEXT DEFAULT CURRENT_TIMESTAMP
)''')

c.execute('''CREATE TABLE IF NOT EXISTS user_interests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    symbol TEXT
)''')

conn.commit()

# ===== إعدادات النظام الذكية =====
class SmartConfig:
    def __init__(self):
        self.POINT_VALUE = 0.001
        self.CONVERSION_FEE = 0.6
        self.MIN_CONVERSION = 5000
        self.MAX_POINTS_PER_USER = 10000
        self.REFERRAL_REWARDS = {1: 500, 2: 250, 3: 100}
        self.REFERRAL_VERIFICATION_DAYS = 3
        self.REFERRAL_BONUS_THRESHOLDS = {5: 200, 10: 500, 20: 1000}
        self.BASE_RECOMMENDATION_PRICE = 10
        self.PRICE_DYNAMICS = {
            'new_user_discount': 0.5,
            'active_user_discount': 0.2,
            'high_balance_penalty': 0.3
        }
        self.COMMISSION_RATE = 0.15
        self.MIN_TRADE_SIZE = 10
        self.MAX_DEVIATION = 0.05
        self.TRUST_SCORE_RANGES = {
            'high': (80, 100),
            'medium': (50, 79),
            'low': (0, 49)
        }
        self.POINT_BURN_RATE = 0.05
        self.ECONOMIC_CHECK_INTERVAL = 604800
        self.DAILY_REWARD = 10
        self.QUESTS = {
            'daily': [
                {"id": "login", "points": 10, "desc": "سجل دخول يومي"},
                {"id": "share", "points": 30, "desc": "شارك البوت"},
                {"id": "audit", "points": 50, "desc": "تدقيق صفقة"}
            ],
            'weekly': [
                {"id": "refer_3", "points": 200, "desc": "أحِل 3 أصدقاء"},
                {"id": "trade_5", "points": 150, "desc": "نفذ 5 صفقات"}
            ]
        }
        self.SUPPORTED_CURRENCIES = ['USD', 'BTC', 'ETH']
        self.MARKET_THRESHOLDS = {
            'bearish': {'fee': 0.5, 'price': 8},
            'bullish': {'fee': 0.7, 'price': 12},
            'normal': {'fee': 0.6, 'price': 10}
        }
        self.FREE_RECOMMENDATIONS = 1
        self.TRIAL_PERIOD = 30
        self.ANALYSTS = [
            "Crypto_Arabia", "Binance_KSA", "CryptoSheikh", "ArabWhale", "EmiratesCrypto",
            "Qatar_Bitcoin", "EgyptBlockchain", "IraqiCrypto", "KuwaitCoin", "Oman_Bitcoin",
            "YemeniTrader", "SyrianCrypto", "JordanCrypto", "LebanonCrypto", "TunisiaBlockchain",
            "AlgeriaCrypto", "MoroccoCrypto", "Sudan_Bitcoin", "LibyaCrypto", "MauritaniaCrypto"
        ]
        self.TOP_100_COINS = [
            "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "DOT", "DOGE", "SHIB",
            "MATIC", "LTC", "TRX", "LINK", "ATOM", "UNI", "XLM", "ALGO", "VET", "ICP"
        ]
        self.CACHE_DIR = "cache"
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self.ADMIN_CHANNEL = "@GoldenPyramidAdmin"
        self.ADMIN_ID = "YOUR_ADMIN_ID"

config = SmartConfig()

# ===== نظام جمع البيانات الذكية =====
class DataCollector:
    def __init__(self, config):
        self.config = config
        self.ua = UserAgent()

    def get_cache_or_fetch(self, key, fetch_function, expiration=3600):
        cache_file = os.path.join(self.config.CACHE_DIR, f"{key}.json")

        if os.path.exists(cache_file):
            modified_time = os.path.getmtime(cache_file)
            if (time.time() - modified_time) < expiration:
                try:
                    with open(cache_file, "r") as f:
                        return json.load(f)
                except:
                    pass

        data = fetch_function()

        if data:
            try:
                with open(cache_file, "w") as f:
                    json.dump(data, f)
            except:
                pass
            return data

        return None

    def fetch_top_coins(self):
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 20,
                "page": 1,
                "sparkline": False
            }
            headers = {'User-Agent': self.ua.random}
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()

            coins = []
            for coin in response.json():
                coins.append({
                    'symbol': coin['symbol'].upper(),
                    'name': coin['name'],
                    'price': coin['current_price']
                })

            return coins

        except Exception as e:
            logger.error(f"خطأ في جلب بيانات العملات: {str(e)}")
            return [
                {"symbol": coin, "name": coin, "price": 0.0} 
                for coin in self.config.TOP_100_COINS[:20]
            ]

    def fetch_analyst_recommendations(self):
        recommendations = []
        today = datetime.now().date()

        for analyst in self.config.ANALYSTS:
            try:
                query = f'from:{analyst} (شراء OR بيع OR احتفاظ) since:{today}'
                for tweet in sntwitter.TwitterSearchScraper(query).get_items():
                    content = tweet.content
                    coins = re.findall(r'\b([A-Z]{3,5})\b', content)

                    recommendations.append({
                        'analyst': analyst,
                        'content': content,
                        'coins': list(set(coins)),
                        'timestamp': tweet.date.isoformat()
                    })
            except Exception as e:
                logger.error(f"خطأ في جلب بيانات {analyst}: {str(e)}")

        return recommendations

# ===== نظام التحليل الذكي =====
class AnalysisEngine:
    def __init__(self, config):
        self.config = config
        self.model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)

    def analyze_data(self, coins_data, recommendations_data):
        try:
            coins = coins_data
            recs = recommendations_data

            analysis_data = []
            for coin in coins:
                symbol = coin['symbol']
                coin_recs = [r for r in recs if symbol in r['coins']]

                if coin_recs:
                    sentiment = self.calculate_sentiment(coin_recs)
                    analysis_data.append({
                        'symbol': symbol,
                        'sentiment': sentiment,
                        'mentions': len(coin_recs)
                    })

            if analysis_data:
                X = np.array([[d['sentiment'], d['mentions']] for d in analysis_data])
                self.model.fit(X)
                predictions = self.model.predict(X)

                recommendations = []
                for i, data in enumerate(analysis_data):
                    if predictions[i] == -1:
                        recommendations.append({
                            'symbol': data['symbol'],
                            'name': next((c['name'] for c in coins if c['symbol'] == data['symbol']), data['symbol']),
                            'sentiment': data['sentiment'],
                            'mentions': data['mentions'],
                            'recommendation': "🚀 شراء قوي" if data['sentiment'] > 0 else "🔻 بيع فوري",
                            'reason': "تحليل إيجابي من المحللين" if data['sentiment'] > 0 else "تحليل سلبي من المحللين"
                        })

                return sorted(recommendations, key=lambda x: abs(x['sentiment']), reverse=True)[:5]

        except Exception as e:
            logger.error(f"خطأ في التحليل: {str(e)}")

        return [{
            'symbol': "BTC",
            'name': "Bitcoin",
            'recommendation': "🚀 شراء قوي",
            'reason': "أداء تاريخي قوي وتحليل إيجابي من المحللين",
            'sentiment': 0.85,
            'mentions': 15
        }]

    def calculate_sentiment(self, recommendations):
        positive_keywords = ["شراء", "ممتاز", "جيد", "يرتفع", "قوي", "فرصة", "مستقبل", "زيادة"]
        negative_keywords = ["بيع", "ضعيف", "هبوط", "تجنب", "خطر", "انخفاض", "مشكلة"]

        total_score = 0
        for rec in recommendations:
            score = 0
            content = rec['content'].lower()

            for word in positive_keywords:
                if word in content:
                    score += 1

            for word in negative_keywords:
                if word in content:
                    score -= 1

            total_score += max(-1, min(1, score / 3))

        return total_score / len(recommendations) if recommendations else 0

# ===== نظام مكافحة التضخم =====
class PointEconomy:
    @staticmethod
    def adjust_supply():
        try:
            c.execute("SELECT SUM(points) FROM users")
            total_points = c.fetchone()[0] or 0

            burn_amount = int(total_points * config.POINT_BURN_RATE)
            c.execute("UPDATE users SET points = points * ?", 
                     (1 - config.POINT_BURN_RATE,))
            conn.commit()

            logger.info(f"تم حرق {burn_amount} نقطة من العرض")
            return burn_amount
        except Exception as e:
            logger.error(f"خطأ في ضبط العرض: {e}")
            return 0

    @staticmethod
    def get_economic_status():
        try:
            c.execute("SELECT SUM(points) FROM users")
            total_circulating = c.fetchone()[0] or 0

            c.execute("SELECT SUM(total_earned) FROM users")
            total_earned = c.fetchone()[0] or 0

            c.execute("SELECT SUM(total_spent) FROM users")
            total_spent = c.fetchone()[0] or 0

            return {
                'circulating': total_circulating,
                'total_earned': total_earned,
                'total_spent': total_spent,
                'health_ratio': total_spent / (total_earned + 1)
            }
        except Exception as e:
            logger.error(f"خطأ في الحصول على الحالة الاقتصادية: {e}")
            return {}

# ===== نظام التحقق =====
class VerificationSystem:
    @staticmethod
    def verify_user(user_id):
        try:
            c.execute("SELECT COUNT(*) FROM transactions WHERE user_id = ?", (user_id,))
            activity_count = c.fetchone()[0]

            if activity_count >= 3:
                c.execute("UPDATE users SET verified = TRUE WHERE user_id = ?", (user_id,))
                conn.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"خطأ في التحقق من المستخدم: {e}")
            return False

    @staticmethod
    def verify_referrals():
        try:
            c.execute('''SELECT referral_id, referrer_id 
                         FROM referrals 
                         WHERE verified = FALSE 
                         AND date < DATE('now', ?)''',
                     (f'-{config.REFERRAL_VERIFICATION_DAYS} days',))

            unverified = c.fetchall()
            for ref_id, referrer_id in unverified:
                if VerificationSystem.verify_user(referrer_id):
                    c.execute("UPDATE referrals SET verified = TRUE WHERE referral_id = ?", (ref_id,))
                    c.execute("SELECT level FROM referrals WHERE referral_id = ?", (ref_id,))
                    level = c.fetchone()[0]
                    reward = config.REFERRAL_REWARDS.get(level, 0)
                    PointSystem.add_points(referrer_id, reward, f"مكافأة إحالة مؤكدة - المستوى {level}")

            conn.commit()
            return len(unverified)
        except Exception as e:
            logger.error(f"خطأ في التحقق من الإحالات: {e}")
            return 0

# ===== نظام النقاط =====
class PointSystem:
    @staticmethod
    def add_points(user_id, amount, reason=""):
        try:
            c.execute("SELECT points FROM users WHERE user_id = ?", (user_id,))
            current = c.fetchone()[0] or 0

            new_total = min(current + amount, config.MAX_POINTS_PER_USER)
            c.execute("UPDATE users SET points = ?, total_earned = total_earned + ? WHERE user_id = ?", 
                     (new_total, amount, user_id))

            c.execute('''INSERT INTO transactions 
                         (user_id, points, type, description)
                         VALUES (?, ?, 'earn', ?)''',
                     (user_id, amount, reason))
            conn.commit()

            return new_total
        except Exception as e:
            logger.error(f"خطأ في إضافة النقاط: {e}")
            return 0

    @staticmethod
    def spend_points(user_id, amount, reason=""):
        try:
            c.execute("SELECT points FROM users WHERE user_id = ?", (user_id,))
            current_points = c.fetchone()[0] or 0

            if current_points >= amount:
                c.execute("UPDATE users SET points = points - ?, total_spent = total_spent + ? WHERE user_id = ?", 
                         (amount, amount, user_id))

                c.execute('''INSERT INTO transactions 
                             (user_id, points, type, description)
                             VALUES (?, ?, 'spend', ?)''',
                         (user_id, amount, reason))
                conn.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"خطأ في خصم النقاط: {e}")
            return False

# ===== نظام التوصيات المتكامل =====
class RecommendationEngine:
    def __init__(self, config):
        self.config = config
        self.collector = DataCollector(config)
        self.analyzer = AnalysisEngine(config)
        self.recommendations = []
        self.last_update = datetime.min

    def get_dynamic_price(self, user_id):
        try:
            base_price = self.config.BASE_RECOMMENDATION_PRICE

            c.execute("SELECT join_date FROM users WHERE user_id = ?", (user_id,))
            join_date = datetime.strptime(c.fetchone()[0], '%Y-%m-%d')
            if (datetime.now() - join_date).days < 3:
                return int(base_price * (1 - self.config.PRICE_DYNAMICS['new_user_discount']))

            c.execute('''SELECT COUNT(*) FROM transactions 
                         WHERE user_id = ? AND date > DATE('now', '-7 days')''',
                     (user_id,))
            weekly_activity = c.fetchone()[0]
            if weekly_activity > 5:
                return int(base_price * (1 - self.config.PRICE_DYNAMICS['active_user_discount']))

            c.execute("SELECT points FROM users WHERE user_id = ?", (user_id,))
            user_points = c.fetchone()[0] or 0
            if user_points > 5000:
                return int(base_price * (1 + self.config.PRICE_DYNAMICS['high_balance_penalty']))

            return base_price
        except Exception as e:
            logger.error(f"خطأ في حساب السعر: {e}")
            return self.config.BASE_RECOMMENDATION_PRICE

    def generate_recommendation(self, user_id, use_free=False):
        try:
            if use_free:
                today = datetime.now().strftime("%Y-%m-%d")
                c.execute("SELECT free_recommendations_used FROM users WHERE user_id = ?", (user_id,))
                free_used = c.fetchone()[0] or 0

                if free_used >= self.config.FREE_RECOMMENDATIONS:
                    return None

                price = 0
            else:
                price = self.get_dynamic_price(user_id)

            if not self.recommendations or (datetime.now() - self.last_update).seconds > 3600:
                coins_data = self.collector.get_cache_or_fetch(
                    "top_coins", 
                    self.collector.fetch_top_coins,
                    expiration=3600
                )
                recs_data = self.collector.get_cache_or_fetch(
                    "analyst_recommendations",
                    self.collector.fetch_analyst_recommendations,
                    expiration=1800
                )
                if coins_data and recs_data:
                    self.recommendations = self.analyzer.analyze_data(coins_data, recs_data)
                    self.last_update = datetime.now()

            if not self.recommendations:
                coins = ["BTC", "ETH", "XRP", "ADA", "SOL", "DOT", "AVAX", "MATIC"]
                actions = ["شراء", "بيع", "احتفاظ"]
                strengths = ["قوية", "متوسطة", "ضعيفة"]

                recommendation = {
                    "symbol": random.choice(coins),
                    "name": random.choice(coins),
                    "recommendation": random.choice(actions),
                    "strength": random.choice(strengths),
                    "confidence": random.randint(70, 95),
                    "reason": "توصية عشوائية"
                }
            else:
                recommendation = random.choice(self.recommendations)

            c.execute('''INSERT INTO recommendations 
                         (user_id, symbol, name, recommendation, reason)
                         VALUES (?, ?, ?, ?, ?)''',
                     (user_id, recommendation['symbol'], recommendation['name'], 
                      recommendation['recommendation'], recommendation.get('reason', '')))

            if not use_free:
                if not PointSystem.spend_points(user_id, price, "شراء توصية"):
                    return None
            else:
                c.execute("UPDATE users SET free_recommendations_used = free_recommendations_used + 1 WHERE user_id = ?", (user_id,))

            conn.commit()

            return recommendation
        except Exception as e:
            logger.error(f"خطأ في توليد التوصية: {e}")
            return None

# ===== نظام الإحالة الهرمي =====
class ReferralSystem:
    @staticmethod
    def add_referral(referrer_id, referred_id):
        try:
            if referrer_id == referred_id:
                return False

            c.execute("SELECT referrer_id FROM users WHERE user_id = ?", (referrer_id,))
            referrer_data = c.fetchone()

            if referrer_data and referrer_data[0]:
                level = 2
                grand_referrer = referrer_data[0]
            else:
                level = 1
                grand_referrer = None

            c.execute('''INSERT INTO referrals 
                         (referrer_id, referred_id, level, date)
                         VALUES (?, ?, ?, CURRENT_TIMESTAMP)''',
                     (referrer_id, referred_id, level))

            if level == 2 and grand_referrer:
                c.execute('''INSERT INTO referrals 
                             (referrer_id, referred_id, level, date)
                             VALUES (?, ?, 3, CURRENT_TIMESTAMP)''',
                         (grand_referrer, referred_id))

            conn.commit()

            PointSystem.add_points(referrer_id, 100, "مكافأة إحالة أولية")
            if grand_referrer:
                PointSystem.add_points(grand_referrer, 50, "مكافأة إحالة غير مباشرة")

            c.execute("SELECT COUNT(*) FROM referrals WHERE referrer_id = ? AND verified = TRUE", 
                     (referrer_id,))
            verified_count = c.fetchone()[0]

            for threshold, bonus in config.REFERRAL_BONUS_THRESHOLDS.items():
                if verified_count >= threshold:
                    PointSystem.add_points(referrer_id, bonus, f"مكافأة عتبة {threshold} إحالة")

            return True
        except Exception as e:
            logger.error(f"خطأ في تسجيل الإحالة: {e}")
            return False

# ===== نظام المهام =====
class QuestSystem:
    @staticmethod
    def assign_quests(user_id):
        try:
            for quest in config.QUESTS['daily']:
                c.execute('''INSERT OR IGNORE INTO quests 
                             (user_id, quest_type, date)
                             VALUES (?, ?, CURRENT_TIMESTAMP)''',
                         (user_id, quest['id']))

            if datetime.now().weekday() == 0:
                for quest in config.QUESTS['weekly']:
                    c.execute('''INSERT OR IGNORE INTO quests 
                                 (user_id, quest_type, date)
                                 VALUES (?, ?, CURRENT_TIMESTAMP)''',
                             (user_id, quest['id']))

            conn.commit()
            return True
        except Exception as e:
            logger.error(f"خطأ في تعيين المهام: {e}")
            return False

    @staticmethod
    def complete_quest(user_id, quest_id):
        try:
            c.execute("SELECT quest_type FROM quests WHERE quest_id = ?", (quest_id,))
            quest_type = c.fetchone()[0]

            c.execute('''UPDATE quests SET progress = progress + 1 
                         WHERE quest_id = ? AND completed = FALSE''',
                     (quest_id,))

            c.execute("SELECT progress FROM quests WHERE quest_id = ?", (quest_id,))
            progress = c.fetchone()[0]

            required = 1
            if quest_type == 'refer_3':
                required = 3
            elif quest_type == 'trade_5':
                required = 5

            if progress >= required:
                c.execute("UPDATE quests SET completed = TRUE WHERE quest_id = ?", (quest_id,))

                for quest in config.QUESTS['daily'] + config.QUESTS['weekly']:
                    if quest['id'] == quest_type:
                        PointSystem.add_points(user_id, quest['points'], f"إكمال مهمة: {quest['desc']}")
                        return quest['points']

            conn.commit()
            return 0
        except Exception as e:
            logger.error(f"خطأ في إكمال المهمة: {e}")
            return 0

# ===== نظام تحصيل الأرباح والتحقق =====
class ProfitCommissionSystem:
    COMMISSION_RATE = 0.15

    @staticmethod
    def report_profit(update: Update, context: CallbackContext, trade_id, profit):
        try:
            user_id = update.effective_user.id

            c.execute("SELECT position_size FROM recommendations WHERE id = ?", (trade_id,))
            position_size = c.fetchone()[0]

            if position_size < config.MIN_TRADE_SIZE:
                update.message.reply_text(
                    f"⚠️ حجم التداول أقل من الحد الأدنى ({config.MIN_TRADE_SIZE} USDT)\n"
                    "لا يمكن تحصيل عمولة لهذه الصفقة"
                )
                return False

            commission = profit * ProfitCommissionSystem.COMMISSION_RATE

            c.execute('''UPDATE recommendations 
                         SET profit = ?, verified = FALSE 
                         WHERE id = ? AND user_id = ?''',
                     (profit, trade_id, user_id))

            c.execute("SELECT trust_score, escrow_balance FROM users WHERE user_id = ?", (user_id,))
            trust_score, escrow_balance = c.fetchone()

            if trust_score >= 80:
                new_balance = escrow_balance - commission
                if new_balance >= 0:
                    c.execute("UPDATE users SET escrow_balance = ? WHERE user_id = ?", 
                             (new_balance, user_id))
                    c.execute("UPDATE recommendations SET verified = TRUE WHERE id = ?", (trade_id,))
                    update.message.reply_text(
                        f"✅ تم تحصيل العمولة: {commission:.2f} USDT\n"
                        f"رصيد الضمان الجديد: {new_balance:.2f} USDT"
                    )
                else:
                    c.execute("UPDATE users SET debt = debt + ? WHERE user_id = ?", 
                             (commission, user_id))
                    update.message.reply_text(
                        f"⚠️ رصيد الضمان غير كافي، تم إضافة دين: {commission:.2f} USDT\n"
                        "الرجاء تعبئة الضمان باستخدام /deposit"
                    )

            elif trust_score >= 50:
                ProfitCommissionSystem.add_to_audit(trade_id)
                update.message.reply_text(
                    f"📝 تم إرسال الصفقة للتدقيق\n"
                    f"العمولة المستحقة: {commission:.2f} USDT\n"
                    "سيتم إعلامك بنتيجة التدقيق"
                )

            else:
                update.message.reply_text(
                    "⚠️ يلزم تحقق إضافي لهذه الصفقة\n\n"
                    "الرجاء إرسال لقطة شاشة من المنصة\n"
                    "باستخدام الأمر: /verify_trade [رقم الصفقة] [رابط الصورة]"
                )

            conn.commit()
            return True

        except Exception as e:
            logger.error(f"خطأ في الإبلاغ عن الأرباح: {e}")
            update.message.reply_text("❌ حدث خطأ أثناء معالجة الطلب")
            return False

    @staticmethod
    def add_to_audit(trade_id):
        try:
            c.execute('''SELECT user_id FROM users 
                         WHERE trust_score >= 80 
                         ORDER BY RANDOM() LIMIT 3''')
            auditors = [row[0] for row in c.fetchall()]

            for auditor_id in auditors:
                c.execute('''INSERT INTO audits 
                             (trade_id, auditor_id, vote)
                             VALUES (?, ?, 'pending')''',
                         (trade_id, auditor_id))

            conn.commit()
            return True
        except Exception as e:
            logger.error(f"خطأ في إضافة الصفقة للتدقيق: {e}")
            return False

    @staticmethod
    def process_audit_vote(update: Update, context: CallbackContext, audit_id, vote):
        try:
            user_id = update.effective_user.id

            c.execute('''UPDATE audits 
                         SET vote = ?, comments = ? 
                         WHERE audit_id = ? AND auditor_id = ?''',
                     (vote, "تصويت بدون تعليق", audit_id, user_id))

            c.execute('''SELECT COUNT(*) FROM audits 
                         WHERE trade_id = (SELECT trade_id FROM audits WHERE audit_id = ?)''',
                     (audit_id,))
            total_votes = c.fetchone()[0]

            c.execute('''SELECT COUNT(*) FROM audits 
                         WHERE trade_id = (SELECT trade_id FROM audits WHERE audit_id = ?) 
                         AND vote = 'approve' ''',
                     (audit_id,))
            approve_votes = c.fetchone()[0]

            if total_votes >= 2:
                trade_id = c.execute('''SELECT trade_id FROM audits 
                                      WHERE audit_id = ?''', (audit_id,)).fetchone()[0]

                if approve_votes >= 2:
                    c.execute('''UPDATE recommendations 
                                 SET verified = TRUE 
                                 WHERE id = ?''', (trade_id,))

                    c.execute("SELECT profit FROM recommendations WHERE id = ?", (trade_id,))
                    profit = c.fetchone()[0]
                    commission = profit * ProfitCommissionSystem.COMMISSION_RATE

                    c.execute("SELECT user_id FROM recommendations WHERE id = ?", (trade_id,))
                    trade_user_id = c.fetchone()[0]

                    c.execute('''UPDATE users 
                                 SET escrow_balance = escrow_balance - ? 
                                 WHERE user_id = ?''', (commission, trade_user_id))

                    for auditor in c.execute('''SELECT auditor_id FROM audits 
                                              WHERE trade_id = ?''', (trade_id,)).fetchall():
                        PointSystem.add_points(auditor[0], 50, "مكافأة تدقيق ناجح")

                    context.bot.send_message(
                        chat_id=trade_user_id,
                        text=f"✅ تم التحقق من صفقتك #{trade_id} وخصم العمولة: {commission:.2f} USDT"
                    )
                else:
                    c.execute('''UPDATE users 
                                 SET trust_score = trust_score - 10 
                                 WHERE user_id = (SELECT user_id FROM recommendations WHERE id = ?)''',
                             (trade_id,))

                    context.bot.send_message(
                        chat_id=trade_user_id,
                        text=f"⚠️ تم رفض صفقتك #{trade_id} بعد التدقيق\n"
                             "تم خفض درجة ثقتك 10 نقاط"
                    )

            conn.commit()
            update.message.reply_text("✅ تم تسجيل تصويتك بنجاح")
            return True

        except Exception as e:
            logger.error(f"خطأ في معالجة التصويت: {e}")
            update.message.reply_text("❌ حدث خطأ أثناء معالجة التصويت")
            return False

# ===== نظام الثقة =====
class TrustSystem:
    @staticmethod
    def update_trust_score(user_id, change):
        try:
            c.execute('''UPDATE users 
                         SET trust_score = MAX(0, MIN(100, trust_score + ?)) 
                         WHERE user_id = ?''',
                     (change, user_id))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"خطأ في تحديث درجة الثقة: {e}")
            return False

    @staticmethod
    def get_trust_level(user_id):
        try:
            c.execute("SELECT trust_score FROM users WHERE user_id = ?", (user_id,))
            score = c.fetchone()[0]

            if score >= 80:
                return "high"
            elif score >= 50:
                return "medium"
            else:
                return "low"
        except:
            return "medium"

# ===== نظام التداول الافتراضي التعليمي =====
class VirtualTradingSystem:
    def __init__(self):
        self.virtual_portfolios = {}

    def create_portfolio(self, user_id, capital=10000):
        self.virtual_portfolios[user_id] = {
            'balance': capital,
            'assets': {},
            'history': []
        }
        return f"🎮 تم إنشاء محفظة افتراضية برأس مال {capital} دولار"

    def execute_trade(self, user_id, symbol, action, amount):
        price = self.get_live_price(symbol)

        portfolio = self.virtual_portfolios.get(user_id)
        if not portfolio:
            return "❌ لم يتم إنشاء محفظة بعد. استخدم /virtual_start"

        cost = amount * price
        if action == 'buy':
            if cost > portfolio['balance']:
                return "❌ رصيد غير كافي"

            portfolio['balance'] -= cost
            portfolio['assets'][symbol] = portfolio['assets'].get(symbol, 0) + amount
        else:
            if portfolio['assets'].get(symbol, 0) < amount:
                return "❌ لا تملك الكمية الكافية"

            portfolio['balance'] += cost
            portfolio['assets'][symbol] -= amount

        portfolio['history'].append({
            'symbol': symbol,
            'action': action,
            'amount': amount,
            'price': price,
            'timestamp': datetime.now()
        })

        return f"✅ تم تنفيذ {action} {amount} من {symbol} بسعر {price}"

    def get_live_price(self, symbol):
        try:
            response = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd")
            return response.json()[symbol.lower()]['usd']
        except:
            return random.uniform(1, 100)

# ===== نظام التحديات المجتمعية الذكية =====
class CommunityChallenges:
    def __init__(self):
        self.active_challenges = {}

    def create_challenge(self, description, target):
        challenge_id = hashlib.md5(description.encode()).hexdigest()[:8]
        self.active_challenges[challenge_id] = {
            'description': description,
            'target': target,
            'progress': 0,
            'participants': {},
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(days=3)
        }
        return challenge_id

    def join_challenge(self, user_id, challenge_id):
        if challenge_id in self.active_challenges:
            self.active_challenges[challenge_id]['participants'][user_id] = 0
            return True
        return False

    def contribute_to_challenge(self, user_id, challenge_id, value):
        if challenge_id in self.active_challenges and user_id in self.active_challenges[challenge_id]['participants']:
            self.active_challenges[challenge_id]['participants'][user_id] += value
            self.active_challenges[challenge_id]['progress'] += value

            if self.active_challenges[challenge_id]['progress'] >= self.active_challenges[challenge_id]['target']:
                self.reward_participants(challenge_id)
                return "challenge_completed"
        return "contribution_added"

    def reward_participants(self, challenge_id):
        challenge = self.active_challenges[challenge_id]
        total_contributions = sum(challenge['participants'].values())

        for user_id, contribution in challenge['participants'].items():
            reward = int(1000 * (contribution / total_contributions))
            PointSystem.add_points(user_id, reward, f"مكافأة تحدي: {challenge['description']}")

        del self.active_challenges[challenge_id]

# ===== نظام إنذار التقلبات الذكي =====
class VolatilityAlertSystem:
    def __init__(self):
        self.price_history = {}

    def monitor_asset(self, symbol):
        current_price = self.get_live_price(symbol)

        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append(current_price)
        if len(self.price_history[symbol]) > 10:
            self.price_history[symbol].pop(0)

        if len(self.price_history[symbol]) >= 5:
            prices = self.price_history[symbol]
            volatility = max(prices) - min(prices)
            avg_price = sum(prices) / len(prices)
            volatility_percent = (volatility / avg_price) * 100

            if volatility_percent > 15:
                return f"⚠️ إنذار تقلب: {symbol} تغير {volatility_percent:.2f}% في آخر 5 قراءات"

        return None

    def get_live_price(self, symbol):
        try:
            response = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd")
            return response.json()[symbol.lower()]['usd']
        except:
            return random.uniform(1, 100)

# ===== نظام الإنجازات الدافع =====
class AchievementSystem:
    def __init__(self):
        self.achievements = {
            'novice_trader': {'name': 'المتداول المبتدئ', 'desc': 'أكمل 5 صفقات', 'target': 5, 'reward': 100},
            'social_butterfly': {'name': 'فراشة اجتماعية', 'desc': 'شارك 10 مرات', 'target': 10, 'reward': 200},
            'audit_expert': {'name': 'خبير التدقيق', 'desc': 'دقق 20 صفقة', 'target': 20, 'reward': 500},
            'risk_taker': {'name': 'محب المخاطرة', 'desc': 'استثمر في 5 عملات مختلفة', 'target': 5, 'reward': 300}
        }

    def check_achievements(self, user_id, action):
        unlocked = []

        c.execute('''INSERT OR IGNORE INTO user_achievements 
                     (user_id, achievement_id, progress) 
                     VALUES (?, ?, 0)''', (user_id, action))

        c.execute('''UPDATE user_achievements 
                     SET progress = progress + 1 
                     WHERE user_id = ? AND achievement_id = ?''',
                 (user_id, action))

        conn.commit()

        for ach_id, ach in self.achievements.items():
            c.execute('''SELECT progress FROM user_achievements 
                         WHERE user_id = ? AND achievement_id = ?''',
                     (user_id, ach_id))
            progress = c.fetchone()

            if progress and progress[0] >= ach['target']:
                c.execute('''SELECT 1 FROM achievements_unlocked 
                             WHERE user_id = ? AND achievement_id = ?''',
                         (user_id, ach_id))
                if not c.fetchone():
                    PointSystem.add_points(user_id, ach['reward'], f"إنجاز: {ach['name']}")
                    c.execute('''INSERT INTO achievements_unlocked 
                                 (user_id, achievement_id) VALUES (?, ?)''',
                             (user_id, ach_id))
                    unlocked.append(ach)

        conn.commit()
        return unlocked

# ===== البوت الرئيسي =====
class GoldenPyramidBot:
    def __init__(self, token):
        self.token = token
        self.updater = Updater(token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.recommendation_engine = RecommendationEngine(config)
        self.commission_system = ProfitCommissionSystem()
        self.virtual_trading = VirtualTradingSystem()
        self.community_challenges = CommunityChallenges()
        self.volatility_alerts = VolatilityAlertSystem()
        self.achievement_system = AchievementSystem()

        # معالجات الأوامر
        self.dispatcher.add_handler(CommandHandler('start', self.start))
        self.dispatcher.add_handler(CommandHandler('buy', self.buy_recommendation))
        self.dispatcher.add_handler(CommandHandler('invite', self.invite_friends))
        self.dispatcher.add_handler(CommandHandler('profile', self.user_profile))
        self.dispatcher.add_handler(CommandHandler('convert', self.convert_points))
        self.dispatcher.add_handler(CommandHandler('quests', self.show_quests))
        self.dispatcher.add_handler(CommandHandler('leaderboard', self.show_leaderboard))
        self.dispatcher.add_handler(CommandHandler('deposit', self.deposit_escrow))
        self.dispatcher.add_handler(CommandHandler('report_profit', self.report_profit))
        self.dispatcher.add_handler(CommandHandler('audit_vote', self.audit_vote))
        self.dispatcher.add_handler(CommandHandler('verify_trade', self.verify_trade))
        self.dispatcher.add_handler(CommandHandler('virtual_start', self.start_virtual_trading))
        self.dispatcher.add_handler(CommandHandler('virtual_trade', self.execute_virtual_trade))
        self.dispatcher.add_handler(CommandHandler('join_challenge', self.join_challenge))
        self.dispatcher.add_handler(CommandHandler('set_interest', self.set_interest))

        # جدولة المهام
        self.updater.job_queue.run_repeating(self.daily_tasks, interval=86400, first=0)
        self.updater.job_queue.run_repeating(self.weekly_tasks, interval=604800, first=0)
        self.updater.job_queue.run_repeating(self.monitor_volatility, interval=300, first=0)

    def start(self, update: Update, context: CallbackContext):
        user = update.effective_user
        referrer_id = None

        if context.args and context.args[0].startswith('ref_'):
            try:
                referrer_id = int(context.args[0].split('_')[1])
            except:
                pass

        if not self.user_exists(user.id):
            referral_code = f"ref_{user.id}"
            c.execute('''INSERT INTO users 
                         (user_id, username, join_date, last_login, referral_code, referrer_id)
                         VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)''',
                     (user.id, user.username, referral_code, referrer_id))

            if referrer_id:
                ReferralSystem.add_referral(referrer_id, user.id)

            conn.commit()
            QuestSystem.assign_quests(user.id)
            PointSystem.add_points(user.id, 100, "مكافأة ترحيبية")

            update.message.reply_text(
                "🎉 مرحباً بك في نظام الهرم الذهبي الذكي!\n\n"
                "🔹 حصلت على 100 نقطة ترحيبية\n"
                "🔹 استخدم /buy لشراء توصياتك الأولى\n"
                "🔹 استخدم /invite لدعوة الأصدقاء وكسب المزيد\n\n"
                f"🔗 رابط إحالتك: https://t.me/bot?start=ref_{user.id}")
        else:
            c.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?", 
                     (user.id,))
            conn.commit()
            PointSystem.add_points(user.id, config.DAILY_REWARD, "مكافأة دخول يومي")

            update.message.reply_text(
                "👋 مرحباً بعودتك! لقد حصلت على 10 نقاط مكافأة.\n"
                "استخدم /buy لشراء توصية جديدة أو /quests لمشاهدة مهامك اليومية.")

    def buy_recommendation(self, update: Update, context: CallbackContext):
        user_id = update.effective_user.id
        recommendation = self.recommendation_engine.generate_recommendation(user_id)

        if recommendation:
            price = self.recommendation_engine.get_dynamic_price(user_id)

            c.execute('''SELECT id FROM recommendations 
                         WHERE user_id = ? 
                         ORDER BY created_at DESC LIMIT 1''',
                     (user_id,))
            trade_id = c.fetchone()[0]

            response = (
                f"💎 توصيتك المميزة (السعر: {price} نقطة):\n\n"
                f"العملة: {recommendation['symbol']} ({recommendation['name']})\n"
                f"الإجراء: {recommendation['recommendation']}\n"
                f"💎 السبب: {recommendation.get('reason', '')}\n\n"
                f"رقم الصفقة: {trade_id}\n"
                f"بعد إغلاق الصفقة، استخدم:\n"
                f"/report_profit {trade_id} [مقدار الربح]\n"
                "لتحصيل العمولة (15% من الربح)"
            )
            update.message.reply_text(response)
        else:
            update.message.reply_text(
                "⚠️ رصيدك من النقاط غير كافي. يمكنك:\n"
                "- انتظار المكافأة اليومية (/profile)\n"
                "- دعوة الأصدقاء (/invite)\n"
                "- إكمال المهام اليومية (/quests)"
            )

    def report_profit(self, update: Update, context: CallbackContext):
        try:
            trade_id = int(context.args[0])
            profit = float(context.args[1])
        except:
            update.message.reply_text("⚠️ استخدام: /report_profit [رقم الصفقة] [مقدار الربح]")
            return

        if profit <= 0:
            update.message.reply_text("⚠️ الربح يجب أن يكون قيمة موجبة")
            return

        if self.commission_system.report_profit(update, context, trade_id, profit):
            self.update_trade_quest(update.effective_user.id)
            self.community_challenges.contribute_to_challenge(
                update.effective_user.id, 
                'profit_challenge', 
                profit
            )
            self.achievement_system.check_achievements(update.effective_user.id, 'trading')

    def audit_vote(self, update: Update, context: CallbackContext):
        try:
            audit_id = int(context.args[0])
            vote = context.args[1].lower()

            if vote not in ['approve', 'reject']:
                update.message.reply_text("⚠️ التصويت يجب أن يكون 'approve' أو 'reject'")
                return

            self.commission_system.process_audit_vote(update, context, audit_id, vote)
            QuestSystem.complete_quest(update.effective_user.id, "audit")
            self.achievement_system.check_achievements(update.effective_user.id, 'auditing')
        except:
            update.message.reply_text("⚠️ استخدام: /audit_vote [رقم التدقيق] [approve/reject]")

    def verify_trade(self, update: Update, context: CallbackContext):
        try:
            trade_id = int(context.args[0])
            proof_url = context.args[1]
        except:
            update.message.reply_text("⚠️ استخدام: /verify_trade [رقم الصفقة] [رابط التحقق]")
            return

        c.execute('''UPDATE recommendations 
                     SET verified = TRUE 
                     WHERE id = ?''', (trade_id,))
        conn.commit()

        TrustSystem.update_trust_score(update.effective_user.id, 5)

        update.message.reply_text(
            "✅ تم التحقق من الصفقة بنجاح\n"
            "تمت إضافة 5 نقاط لدرجة ثقتك"
        )

    def deposit_escrow(self, update: Update, context: CallbackContext):
        user_id = update.effective_user.id

        try:
            amount = float(context.args[0])
        except:
            update.message.reply_text("⚠️ استخدام: /deposit [المبلغ]")
            return

        c.execute("SELECT escrow_balance FROM users WHERE user_id = ?", (user_id,))
        current_balance = c.fetchone()[0] or 0

        new_balance = current_balance + amount
        c.execute("UPDATE users SET escrow_balance = ? WHERE user_id = ?", 
                 (new_balance, user_id))

        c.execute("SELECT debt FROM users WHERE user_id = ?", (user_id,))
        debt = c.fetchone()[0] or 0

        if debt > 0:
            deduction = min(new_balance, debt)
            new_balance -= deduction
            debt -= deduction

            c.execute("UPDATE users SET escrow_balance = ?, debt = ? WHERE user_id = ?", 
                     (new_balance, debt, user_id))

            update.message.reply_text(
                f"✅ تم إيداع {amount} USDT في الضمان\n"
                f"تم خصم {deduction} USDT لسداد الدين\n"
                f"رصيد الضمان الجديد: {new_balance:.2f} USDT\n"
                f"الدين المتبقي: {debt:.2f} USDT"
            )
        else:
            update.message.reply_text(
                f"✅ تم إيداع {amount} USDT في الضمان\n"
                f"رصيد الضمان الجديد: {new_balance:.2f} USDT"
            )

        TrustSystem.update_trust_score(user_id, 3)
        conn.commit()

    def invite_friends(self, update: Update, context: CallbackContext):
        user_id = update.effective_user.id
        try:
            c.execute("SELECT referral_code FROM users WHERE user_id = ?", (user_id,))
            referral_code = c.fetchone()[0]

            c.execute('''SELECT COUNT(*) FROM referrals 
                         WHERE referrer_id = ? AND verified = TRUE''', 
                     (user_id,))
            verified_count = c.fetchone()[0]

            potential_rewards = sum(config.REFERRAL_REWARDS.values())

            next_threshold = None
            next_bonus = 0
            for threshold, bonus in sorted(config.REFERRAL_BONUS_THRESHOLDS.items()):
                if verified_count < threshold:
                    next_threshold = threshold
                    next_bonus = bonus
                    break

            message = (
                f"📣 رابط دعوتك:\n"
                f"https://t.me/bot?start=ref_{user_id}\n\n"
                f"🔹 عدد الإحالات المؤكدة: {verified_count}\n"
                f"🔹 المكافآت المحتملة: {potential_rewards} نقطة\n"
                f"🔹 المكافآت المتراكمة: {verified_count * 100} نقطة\n\n"
            )

            if next_threshold:
                needed = next_threshold - verified_count
                message += (
                    f"🎯 احصل على مكافأة {next_bonus} نقطة عند وصولك إلى "
                    f"{next_threshold} إحالات مؤكدة (تحتاج {needed} إحالات)"
                )

            keyboard = [
                [InlineKeyboardButton("مشاركة الرابط", switch_inline_query=f"انضم عبر رابط الإحالة الخاص بي: https://t.me/bot?start=ref_{user_id}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            update.message.reply_text(message, reply_markup=reply_markup)
        except Exception as e:
            logger.error(f"خطأ في عرض رابط الإحالة: {e}")
            update.message.reply_text("❌ حدث خطأ أثناء معالجة الطلب")

    def user_profile(self, update: Update, context: CallbackContext):
        user_id = update.effective_user.id
        try:
            c.execute('''SELECT username, points, level, total_earned, total_spent, 
                         trust_score, escrow_balance, debt, free_recommendations_used 
                         FROM users WHERE user_id = ?''', 
                     (user_id,))
            data = c.fetchone()

            username, points, level, total_earned, total_spent, trust_score, escrow, debt, free_used = data

            usd_value = points * config.POINT_VALUE

            trust_level = TrustSystem.get_trust_level(user_id)
            trust_emoji = "🟢" if trust_level == "high" else "🟡" if trust_level == "medium" else "🔴"

            free_remaining = max(0, config.FREE_RECOMMENDATIONS - free_used)

            message = (
                f"👤 الملف الشخصي: @{username}\n"
                f"🏆 المستوى: {level}\n"
                f"⭐ النقاط: {points} (≈${usd_value:.2f})\n"
                f"💳 الضمان: ${escrow:.2f}\n"
                f"📉 الدين: ${debt:.2f}\n\n"
                f"📊 الإحصائيات:\n"
                f"▫️ إجمالي الربح: {total_earned} نقطة\n"
                f"▫️ إجمالي الصرف: {total_spent} نقطة\n"
                f"▫️ توصيات مجانية متبقية: {free_remaining}/{config.FREE_RECOMMENDATIONS}\n"
                f"▫️ درجة الثقة: {trust_score} {trust_emoji}\n\n"
                f"استخدم /convert لتحويل النقاط إلى أموال"
            )

            update.message.reply_text(message)
        except Exception as e:
            logger.error(f"خطأ في عرض الملف الشخصي: {e}")
            update.message.reply_text("❌ حدث خطأ أثناء معالجة الطلب")

    def convert_points(self, update: Update, context: CallbackContext):
        user_id = update.effective_user.id
        try:
            amount = int(context.args[0]) if context.args else 0

            if amount < config.MIN_CONVERSION:
                update.message.reply_text(
                    f"⚠️ الحد الأدنى للتحويل هو {config.MIN_CONVERSION} نقطة\n"
                    f"قيمة النقطة: ${config.POINT_VALUE:.4f}"
                )
                return

            conversion_fee = amount * config.CONVERSION_FEE
            net_amount = amount - conversion_fee
            usd_value = net_amount * config.POINT_VALUE

            if PointSystem.spend_points(user_id, amount, "تحويل نقاط"):
                message = (
                    f"✅ تم تحويل {amount} نقطة بنجاح\n\n"
                    f"▫️ القيمة الصافية: ${usd_value:.2f}\n"
                    f"▫️ الرسوم: {conversion_fee} نقطة ({config.CONVERSION_FEE*100}%)\n\n"
                    "سيتم تحويل الأموال إلى محفظتك خلال 1-3 أيام عمل"
                )
                TrustSystem.update_trust_score(user_id, 2)
            else:
                message = "❌ رصيدك النقطي غير كافي للتحويل"

            update.message.reply_text(message)
        except Exception as e:
            logger.error(f"خطأ في تحويل النقاط: {e}")
            update.message.reply_text("⚠️ استخدام: /convert [عدد النقاط]")

    def show_quests(self, update: Update, context: CallbackContext):
        user_id = update.effective_user.id
        try:
            c.execute('''SELECT quest_id, quest_type, progress, completed 
                         FROM quests 
                         WHERE user_id = ? AND completed = FALSE''', 
                     (user_id,))
            quests = c.fetchall()

            if not quests:
                update.message.reply_text("🎉 لا توجد مهام نشطة حالياً! عد لاحقاً.")
                return

            message = "📋 مهامك اليومية والأسبوعية:\n\n"
            for quest_id, quest_type, progress, completed in quests:
                quest_desc = "مهمة غير معروفة"
                required = 1

                for q in config.QUESTS['daily'] + config.QUESTS['weekly']:
                    if q['id'] == quest_type:
                        quest_desc = q['desc']
                        if '3' in quest_desc:
                            required = 3
                        elif '5' in quest_desc:
                            required = 5
                        break

                progress_bar = "🟩" * progress + "⬜" * (required - progress)
                message += (
                    f"▫️ {quest_desc}\n"
                    f"   التقدم: {progress}/{required} {progress_bar}\n"
                    f"   المعرف: {quest_id}\n\n"
                )

            message += "أكمل المهام لتحصل على نقاط إضافية!"
            update.message.reply_text(message)
        except Exception as e:
            logger.error(f"خطأ في عرض المهام: {e}")
            update.message.reply_text("❌ حدث خطأ أثناء معالجة الطلب")

    def show_leaderboard(self, update: Update, context: CallbackContext):
        try:
            c.execute('''SELECT username, points 
                         FROM users 
                         ORDER BY points DESC 
                         LIMIT 10''')
            top_users = c.fetchall()

            message = "🏆 أفضل 10 أعضاء حسب النقاط:\n\n"
            for i, (username, points) in enumerate(top_users, 1):
                usd_value = points * config.POINT_VALUE
                message += f"{i}. @{username}: {points} نقطة (≈${usd_value:.2f})\n"

            user_id = update.effective_user.id
            c.execute("SELECT points FROM users WHERE user_id = ?", (user_id,))
            user_points = c.fetchone()[0] or 0

            c.execute("SELECT COUNT(*) FROM users WHERE points > ?", (user_points,))
            user_rank = c.fetchone()[0] + 1

            message += (
                f"\n📊 ترتيبك الحالي: #{user_rank}\n"
                f"نقاطك: {user_points} (≈${user_points * config.POINT_VALUE:.2f})"
            )

            update.message.reply_text(message)
        except Exception as e:
            logger.error(f"خطأ في عرض لوحة المتصدرين: {e}")
            update.message.reply_text("❌ حدث خطأ أثناء معالجة الطلب")

    def start_virtual_trading(self, update: Update, context: CallbackContext):
        user_id = update.effective_user.id
        response = self.virtual_trading.create_portfolio(user_id)
        update.message.reply_text(response)

    def execute_virtual_trade(self, update: Update, context: CallbackContext):
        try:
            symbol = context.args[0].upper()
            action = context.args[1].lower()
            amount = float(context.args[2])
        except:
            update.message.reply_text("⚠️ استخدام: /virtual_trade [الرمز] [buy/sell] [الكمية]")
            return

        user_id = update.effective_user.id
        response = self.virtual_trading.execute_trade(user_id, symbol, action, amount)
        self.achievement_system.check_achievements(user_id, 'trading')
        update.message.reply_text(response)

    def join_challenge(self, update: Update, context: CallbackContext):
        try:
            challenge_id = context.args[0]
        except:
            update.message.reply_text("⚠️ استخدام: /join_challenge [معرف التحدي]")
            return

        user_id = update.effective_user.id
        if self.community_challenges.join_challenge(user_id, challenge_id):
            update.message.reply_text("✅ انضممت للتحدي بنجاح!")
        else:
            update.message.reply_text("❌ لم يتم العثور على التحدي")

    def set_interest(self, update: Update, context: CallbackContext):
        try:
            symbol = context.args[0].upper()
            user_id = update.effective_user.id

            c.execute("DELETE FROM user_interests WHERE user_id = ?", (user_id,))
            c.execute("INSERT INTO user_interests (user_id, symbol) VALUES (?, ?)", 
                     (user_id, symbol))
            conn.commit()

            update.message.reply_text(f"✅ تم تعيين اهتمامك لمراقبة: {symbol}")
        except:
            update.message.reply_text("⚠️ استخدام: /set_interest [رمز العملة]")

    def monitor_volatility(self, context: CallbackContext):
        symbols = ['BTC', 'ETH', 'XRP', 'ADA', 'SOL']

        for symbol in symbols:
            alert = self.volatility_alerts.monitor_asset(symbol)
            if alert:
                c.execute("SELECT user_id FROM user_interests WHERE symbol = ?", (symbol,))
                for (user_id,) in c.fetchall():
                    context.bot.send_message(chat_id=user_id, text=alert)

    def daily_tasks(self, context: CallbackContext):
        logger.info("تشغيل المهام اليومية")
        VerificationSystem.verify_referrals()

        c.execute("SELECT user_id FROM users WHERE last_login > DATE('now', '-3 days')")
        for (user_id,) in c.fetchall():
            QuestSystem.assign_quests(user_id)

    def weekly_tasks(self, context: CallbackContext):
        logger.info("تشغيل المهام الأسبوعية")
        PointEconomy.adjust_supply()
        status = PointEconomy.get_economic_status()
        logger.info(f"الحالة الاقتصادية: {status}")

    def update_trade_quest(self, user_id):
        c.execute('''UPDATE quests SET progress = progress + 1 
                     WHERE user_id = ? AND quest_type = 'trade_5' AND completed = FALSE''',
                 (user_id,))
        conn.commit()

        c.execute('''SELECT quest_id FROM quests 
                     WHERE user_id = ? AND quest_type = 'trade_5' 
                     AND progress >= 5 AND completed = FALSE''',
                 (user_id,))
        quest = c.fetchone()
        if quest:
            QuestSystem.complete_quest(user_id, quest[0])

    def user_exists(self, user_id):
        c.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
        return c.fetchone() is not None

    def run(self):
        self.updater.start_polling()
        logger.info("بدأ البوت في الاستماع للأوامر...")
        self.updater.idle()

# ===== التشغيل =====
if __name__ == '__main__':
    PointEconomy.adjust_supply()
    bot = GoldenPyramidBot("YOUR_TELEGRAM_BOT_TOKEN")
    bot.run()