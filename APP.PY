"""
A customer service agent for Kenting Breeze B&B (Gradio app) with RAG.

Every question goes to the agent first. The agent decides which tool to use:
- Basic B&B facts and opening hours -> get_bnb_info (SQLite)
- Room and service prices/availability -> get_room_service (SQLite)
- Stay total calculation with discount -> calculate_stay_cost (Code logic)
- Shop rules and detailed policies -> search_shop_rules (RAG Semantic Search)
- Complaints or unanswerable queries -> transfer_to_human (Escalation)

Run it with:  python app.py
"""

import csv
import json
import os
import re
from pathlib import Path
import sqlite3
import numpy as np
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. API Setup (Chat Model & Embedding Model)
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent / ".env" if "__file__" in locals() else ".env")

CHAT_MODEL = os.getenv("MODEL_NAME", "gemini-3.5-flash-lite")
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("LLM_API_URL"),
    timeout=60.0
)

# Embedding Client: 若未特別設定則 fallback 至主伺服器
EMBED_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
embed_client = OpenAI(
    api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("API_KEY"),
    base_url=os.getenv("EMBEDDING_API_URL") or os.getenv("LLM_API_URL"),
    timeout=60.0
)

# ---------------------------------------------------------------------------
# 2. Database Setup: room_info and room_services (SQLite)
# ---------------------------------------------------------------------------

connection = sqlite3.connect("breeze_bnb_app.db", check_same_thread=False)
connection.row_factory = sqlite3.Row
cursor = connection.cursor()

cursor.execute("DROP TABLE IF EXISTS room_info")
cursor.execute("CREATE TABLE room_info (topic TEXT PRIMARY KEY, content TEXT NOT NULL)")

cursor.execute("DROP TABLE IF EXISTS room_services")
cursor.execute("""CREATE TABLE room_services (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    price INTEGER NOT NULL,
    available INTEGER NOT NULL
)""")

# 讀取 room_info.csv
room_info_file = Path(__file__).parent / "room_info.csv" if "__file__" in locals() else Path("room_info.csv")
if room_info_file.exists():
    with open(room_info_file, newline="", encoding="utf-8") as f:
        cursor.executemany(
            "INSERT INTO room_info (topic, content) VALUES (?, ?)",
            [(r["topic"], r["content"]) for r in csv.DictReader(f)]
        )

# 讀取 rooms.csv 或 room.csv
rooms_file = Path(__file__).parent / "rooms.csv" if "__file__" in locals() else Path("rooms.csv")
if not rooms_file.exists():
    rooms_file = Path(__file__).parent / "room.csv" if "__file__" in locals() else Path("room.csv")

if rooms_file.exists():
    with open(rooms_file, newline="", encoding="utf-8") as f:
        records = []
        for r in csv.DictReader(f):
            is_avail = 1 if str(r["available"]).strip().lower() in ["true", "1"] else 0
            records.append((r["name"], int(r["price"]), is_avail))

        cursor.executemany(
            "INSERT INTO room_services (name, price, available) VALUES (?, ?, ?)",
            records
        )
connection.commit()

BNB_TOPICS = [r["topic"] for r in cursor.execute("SELECT topic FROM room_info")]
print(f"Ready: {len(BNB_TOPICS)} B&B topics loaded into SQLite.")

# ---------------------------------------------------------------------------
# 3. RAG Setup: Shop Rules Chunking & Vector Indexing
# ---------------------------------------------------------------------------

shop_rules_path = Path(__file__).parent / "shop_rules.txt" if "__file__" in locals() else Path("shop_rules.txt")

# 若無 shop_rules.txt 則自動建立範本規章
if not shop_rules_path.exists():
    default_rules = """## 1. 生日派對與預訂規定 Birthday Party & Reservation Policy
預訂生日派對需支付 NT$1,000 訂金。活動前 3 小時取消視為晚退訂；活動 3 小時前取消可退還訂金。一年內累積三次晚退訂者，未來僅接受電話預訂。

## 2. 寵物攜帶規定 Pet Policy
本店家為寵物友善空間（限小型犬），每隻寵物加收清潔費 NT$500。攜帶寵物進店需提前告知，且必須使用牽繩或放入寵物籃。

## 3. 失物招領處理 Lost & Found Policy
顧客遺失物品將由店家妥善保管 30 天。若超過 30 天未有人前來領取，將移交當地警察局處理或依規定處置。

## 4. Wi-Fi與內用時間限制 Wi-Fi & Seating Policy
全館提供免費 Wi-Fi，帳號密碼請參閱內用桌牌。使用 Wi-Fi 與內用空間無時間限制，歡迎安心學習與用餐。

## 5. 外帶與禁帶外食規定 Outside Food & Beverage Policy
店內禁止攜帶外食與外部飲料。若為慶生活動攜帶外帶蛋糕，需加收 NT$200 清潔費。

## 6. 付款與發票載具 Payment & E-Invoice Policy
店內支援現金、信用卡、LINE Pay 與街口支付。結帳時可主動提供手機條碼儲存雲端發票，發票開立後恕不更換。

## 7. 低消與服務費說明 Minimum Charge & Service Charge
每人內用最低消費為任意一份餐點或 NT$100 以上飲料。店內內用與外帶均不收取額外 10% 服務費。

## 8. 場地租借與包場規定 Venue Rental & Private Events
提供包場與會議場地租借服務，需於至少 7 天前提出申請並支付 50% 訂金。活動前 3 天內取消包場者，訂金不予退還。

## 9. 兒童與安全設施 Child Policy & Safety
店家提供兒童座椅與兒童餐具，請家長留意孩童安全，請勿在店內走道跑跳衝撞，以免發生燙傷或碰撞意外。

## 10. 吸菸與違禁品限制 Smoking & Prohibited Items
全館內用與外圍區域全面禁菸（包含電子菸）。店內嚴禁攜帶任何危險物品或違禁品，違者將立即通報警察機關處理。"""
    with open(shop_rules_path, "w", encoding="utf-8") as f:
        f.write(default_rules.strip())

with open(shop_rules_path, "r", encoding="utf-8") as f:
    raw_rules_content = f.read()

# 切分為 Chunk
if raw_rules_content.startswith("## "):
    parts = raw_rules_content[3:].split("\n## ")
else:
    parts = raw_rules_content.split("\n## ")[1:]

chunks = []
for part in parts:
    title, _, text = part.partition("\n")
    chunks.append({"title": title.strip(), "text": text.strip()})

def embed(texts):
    """將文字列表轉為 2D NumPy 陣列 (N x 向量維度)"""
    resp = embed_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([item.embedding for item in resp.data], dtype="float32")

# 建立店內規章的向量索引（Phase A）
chunk_vectors = embed([f"{c['title']}\n{c['text']}" for c in chunks])
MIN_RAG_SCORE = 0.45
print(f"RAG Index Ready: {chunk_vectors.shape[0]} chunks embedded.")

# ---------------------------------------------------------------------------
# 4. Agent Tools Implementation
# ---------------------------------------------------------------------------

def get_bnb_info(topic):
    """Return one piece of information about Kenting Breeze B&B."""
    row = cursor.execute("SELECT content FROM room_info WHERE topic = ?", (topic.strip().lower(),)).fetchone()
    if row is None:
        return {"found": False, "topics_available": BNB_TOPICS}
    return {"found": True, "topic": topic, "content": row["content"]}


def get_room_service(item_name):
    """Find room types, tickets, transport, or rental services and return prices and availability."""
    like = f"%{item_name.strip()}%"
    rows = cursor.execute(
        "SELECT name, price, available FROM room_services WHERE name LIKE ?",
        (like,)
    ).fetchall()
    if not rows:
        return {"found": False, "note": f"查無包含 '{item_name}' 的房型或服務項目。"}
    return {
        "found": True,
        "items": [{
            "name": r["name"],
            "price": r["price"],
            "available": bool(r["available"])
        } for r in rows]
    }


def calculate_stay_cost(items, consecutive_nights: int = 1):
    """Calculate total cost for room reservations and add-on services with consecutive stay discount."""
    lines, raw_total = [], 0
    discount = 0

    for entry in items:
        found = get_room_service(entry["name"])
        if not found["found"] or len(found["items"]) != 1:
            matching_names = [it["name"] for it in found.get("items", [])]
            return {
                "success": False,
                "note": f"無法確認品項 '{entry['name']}'（匹配項目：{matching_names}）。請使用精確名稱或先以 get_room_service 查詢。"
            }

        item = found["items"][0]
        if not item["available"]:
            return {
                "success": False,
                "note": f"項目 '{item['name']}' 目前客滿或暫停供應，無法加入計算。"
            }

        qty = entry["quantity"]
        subtotal = item["price"] * qty
        raw_total += subtotal

        is_room = any(keyword in item["name"] for keyword in ["木屋", "雙人房", "合院", "家庭"])
        if is_room and consecutive_nights >= 2:
            item_discount = int(subtotal * 0.05)
            discount += item_discount
            lines.append(f"{qty} x {item['name']} = NT${subtotal}（享連住 95 折，折抵 NT${item_discount}）")
        else:
            lines.append(f"{qty} x {item['name']} = NT${subtotal}")

    final_total = raw_total - discount
    return {
        "success": True,
        "details": lines,
        "consecutive_nights": consecutive_nights,
        "original_total": raw_total,
        "discount_saved": discount,
        "final_total": final_total,
        "currency": "NT$"
    }


def search_shop_rules(question: str, k: int = 2):
    """Find the chunks of the shop rules closest to the question using semantic vector search."""
    q_vec = embed([question])[0]
    scores = (
        chunk_vectors
        @ q_vec
        / (np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(q_vec))
    )

    best_idx = [i for i in np.argsort(scores)[::-1][:k] if scores[i] >= MIN_RAG_SCORE]
    if not best_idx:
        return {
            "found": False,
            "note": "店內規定規章中未包含此問題的相關條款。"
        }

    return {
        "found": True,
        "chunks": [
            {
                "title": chunks[i]["title"],
                "text": chunks[i]["text"],
                "score": round(float(scores[i]), 3),
            }
            for i in best_idx
        ],
    }


def transfer_to_human():
    """Hand the conversation to human staff for complaints, unanswerable questions, or manager requests."""
    return {
        "message": "請致電民宿現場服務櫃台專線 08-886-1234（服務時間為每日 08:30 至 21:30），或於現場向櫃台人員反映。",
        "note": (
            "The hand-over is done. First politely apologize or explain why in one sentence, "
            "then provide this contact message. Do not say that you will directly transfer or connect the call, "
            "because you cannot. Do not call this tool again."
        )
    }


TOOL_FUNCTIONS = {
    "get_bnb_info": get_bnb_info,
    "get_room_service": get_room_service,
    "calculate_stay_cost": calculate_stay_cost,
    "search_shop_rules": search_shop_rules,
    "transfer_to_human": transfer_to_human,
}

# ---------------------------------------------------------------------------
# 5. Tool Specs & Rules Configuration
# ---------------------------------------------------------------------------

topics_description = ", ".join(BNB_TOPICS) if BNB_TOPICS else "營業時間、停車、早餐、位置等基礎設定"
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_bnb_info",
            "description": f"查詢店家基本資訊，包含基礎設施與服務。可用主題: {topics_description}。",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "enum": BNB_TOPICS, "description": "查詢主題。"}
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_room_service",
            "description": "查詢房型、加購服務、交通票券之價格與供應狀態。請輸入簡短關鍵字，如：'觀海', '雙人', '烤肉', '接送'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string", "description": "欲查詢的房型或服務名稱關鍵字。"}
                },
                "required": ["item_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_stay_cost",
            "description": "計算住宿預訂與加購服務之總費用。若連續入住同房型 2 晚以上可享有 95 折優惠。所有費用計算必須使用此工具，不得自行心算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "房型或加購品項清單。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "資料庫中的精確品項名稱。"},
                                "quantity": {"type": "integer", "description": "預訂數量或夜數。"}
                            },
                            "required": ["name", "quantity"]
                        }
                    },
                    "consecutive_nights": {
                        "type": "integer",
                        "description": "連續住宿夜數（預設 1）。大於等於 2 晚即享有折扣。"
                    }
                },
                "required": ["items"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_shop_rules",
            "description": (
                "搜尋店內規定與條款，包含：寵物攜帶與限制 (pet policy)、生日派對與訂金退訂 (party bookings/deposits)、"
                "失物招領 (lost & found)、Wi-Fi 與用餐/內用時間限制、外帶與外食清潔費、付款方式與發票、低消說明、包場服務、兒童安全與禁菸規定。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "顧客關於店內規章、條款或限制的具體提問字句。"
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human",
            "description": "當發生客訴、要求找店長/經理，或 search_shop_rules 及其他工具均無法回答時使用。",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]

SYSTEM_PROMPT = (
    "You are the customer service agent of Kenting Breeze B&B (墾丁微風海景民宿).\n"
    "All prices are in New Taiwan dollars, written like NT$3,200.\n\n"
    "RULES\n"
    "1. Answer only from tool results. Never answer from memory, and never invent policies, rates, or rules.\n"
    "2. Basic shop info (check-in times, breakfast, parking) -> get_bnb_info.\n"
    "3. Room rates, availability, and extra service prices -> get_room_service.\n"
    "4. Total calculation and multi-night discounts -> calculate_stay_cost. Never calculate numbers yourself.\n"
    "5. Shop rules (party booking, deposit refund, cancellation, pet policy, lost & found, Wi-Fi limit, outside food fee, venue rental) -> search_shop_rules.\n"
    "6. When answering from shop rules, end the answer with the chunk title: (source: 規章標題).\n"
    "7. If no tool can answer, or for complaints and manager requests -> transfer_to_human.\n"
    "8. Only talk about this B&B. Answer politely in the customer's language, in three sentences or less."
)

MAX_STEPS = 6

# ---------------------------------------------------------------------------
# 6. Agent Loop Execution
# ---------------------------------------------------------------------------

def run_agent(chat_history):
    """Answer the last question in chat_history. Returns (answer, trace)."""
    last_msg = chat_history[-1]
    raw_content = last_msg.get("content", "") if isinstance(last_msg, dict) else last_msg[0]

    if isinstance(raw_content, list):
        question = " ".join([item if isinstance(item, str) else item.get("text", str(item)) for item in raw_content])
    elif isinstance(raw_content, dict):
        question = raw_content.get("text", str(raw_content))
    else:
        question = str(raw_content)

    language = "Traditional Chinese" if re.search(r"[\u4e00-\u9fff]", question) else "English"

    clean_history = []
    for msg in chat_history[:-1]:
        if isinstance(msg, dict):
            c = msg.get("content", "")
            if isinstance(c, list):
                c = " ".join([i if isinstance(i, str) else i.get("text", str(i)) for i in c])
            clean_history.append({"role": msg.get("role", "user"), "content": str(c)})
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            u, b = msg
            if u:
                clean_history.append({"role": "user", "content": str(u)})
            if b:
                clean_history.append({"role": "assistant", "content": str(b)})

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + clean_history
        + [{"role": "system", "content": f"Reply in {language}."}]
        + [{"role": "user", "content": question}]
    )
    trace = []

    for step in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOL_SPECS,
            temperature=0.2,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return (message.content or "").strip(), trace

        messages.append(message)

        for call in message.tool_calls:
            arguments = json.loads(call.function.arguments or "{}")
            if call.function.name in TOOL_FUNCTIONS:
                try:
                    result = TOOL_FUNCTIONS[call.function.name](**arguments)
                except Exception as e:
                    result = {"status": "error", "message": f"Execution error: {str(e)}"}
            else:
                result = {"status": "error", "message": f"Tool '{call.function.name}' not found."}

            trace.append({
                "step": step + 1,
                "tool": call.function.name,
                "arguments": arguments,
                "result": result
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })

    return "抱歉，目前系統處理逾時，請直接撥打民宿櫃台專線 08-886-1234 由專人為您服務。", trace

# ---------------------------------------------------------------------------
# 7. Gradio User Interface
# ---------------------------------------------------------------------------

GREETING = (
    "Hello! Welcome to Kenting Breeze B&B (墾丁微風海景民宿).\n"
    "Feel free to ask about room rates, shop rules, check-in policies, or calculate your stay cost!\n\n"
    "您好！我是墾丁微風海景民宿的線上客服小幫手，很高興為您服務！\n"
    "您可以向我詢問房價、入住退房時間、店內規定（派對、寵物、失物、外食）或請我試算入住總費用。"
)

EMPTY_TRACE = "### Tools used\n\n(nothing yet)"

SAMPLE_GROUPS_ZH = {
    "店內規章與政策檢索（search_shop_rules / RAG）": [
        "預訂生日派對如果活動前 2 小時才取消，訂金會退還嗎？",
        "請問生日派對可以自己帶外帶蛋糕進來慶生嗎？需要另外收費嗎？",
        "我昨天退房時把錢包忘在店裡，你們通常會保管多久？",
        "請問店內有提供 Wi-Fi 嗎？有內用時間限制嗎？",
        "請問店內可以帶貓狗等寵物嗎？清潔費是多少？",
    ],
    "房型與加購服務查詢（get_room_service）": [
        "請問觀海雙人木屋平日和假日一晚多少錢？",
        "尊榮VIP六人合院假日還有空房嗎？",
        "高鐵左營站有接送服務嗎？費用多少？",
    ],
    "住宿與加購費用試算（calculate_stay_cost）": [
        "我們想訂觀海雙人木屋平日住 2 晚，再加上 2 張水上活動套票和租一台電動機車，這樣總共多少錢？",
    ],
    "客訴與專人處理（transfer_to_human）": [
        "昨天入住時冷氣一直漏水而且有異味，我要找你們經理投訴！",
        "有賣披薩或珍珠奶茶嗎？",
    ],
}

SAMPLE_GROUPS = {
    "Shop Rules & Policies (search_shop_rules / RAG)": [
        "Can I cancel my birthday party two hours before?",
        "Can we bring our own birthday cake to the party?",
        "I forgot my phone here yesterday, how long do you keep lost items?",
        "Is there free Wi-Fi and any seating time limit?",
    ],
    "Room Rates & Availability (get_room_service)": [
        "How much is the Ocean View Double Cabin on weekdays and weekends?",
        "Do you have shuttle service from Zuoying THSR Station?",
    ],
    "Stay Cost Calculation (calculate_stay_cost)": [
        "I'd like to book 2 nights for Ocean View Double Cabin on weekdays, plus 2 water activity tickets. How much is it?",
    ],
    "Human Transfer & Other": [
        "The room had an odor during my stay. I want to speak to the manager.",
    ],
}

def format_trace(trace):
    if not trace:
        return "### Tools used\n\n(no tool was needed for this answer)"
    lines = ["### Tools used", ""]
    for entry in trace:
        lines += [
            f"**{entry['step']}. {entry['tool']}**",
            "",
            f"arguments: `{json.dumps(entry['arguments'], ensure_ascii=False)[:160]}`",
            "",
            f"result: `{json.dumps(entry['result'], ensure_ascii=False)[:260]}`",
            "",
        ]
    return "\n".join(lines)

def start_chat():
    return [{"role": "assistant", "content": GREETING}]

def add_user_message(user_text, chat_history):
    user_text = user_text.strip()
    if not user_text:
        return "", chat_history
    return "", chat_history + [{"role": "user", "content": user_text}]

def answer_question(chat_history):
    if not chat_history:
        return chat_history, EMPTY_TRACE
    last_msg = chat_history[-1]
    is_user = (isinstance(last_msg, dict) and last_msg.get("role") == "user") or (isinstance(last_msg, (list, tuple)) and last_msg[0] and not last_msg[1])
    if not is_user:
        return chat_history, EMPTY_TRACE
    try:
        answer, trace = run_agent(chat_history)
        panel = format_trace(trace)
    except Exception as error:
        answer = f"(系統訊息) API 呼叫異常: {error}"
        panel = "### Tools used\n\n(the call failed)"
    return chat_history + [{"role": "assistant", "content": answer}], panel

with gr.Blocks(title="墾丁微風海景民宿客服 Agent (RAG)") as demo:
    gr.Markdown(
        "## 墾丁微風海景民宿 (Kenting Breeze B&B) 線上客服小幫手\n"
        "本系統整合 **SQLite 資料庫**（基本資訊、房價查詢）、**Python 業務計算**（連住折扣試算）、"
        "**語意向量檢索 RAG**（店內規章與例外條款）及 **真人轉接機制**。"
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(value=start_chat(), height=450, label="民宿客服助理")
            with gr.Row():
                user_box = gr.Textbox(
                    placeholder="請詢問店內規章、房型房價或費用試算...",
                    show_label=False,
                    scale=8,
                )
                send_button = gr.Button("發送", variant="primary", scale=1)
                clear_button = gr.Button("清除", scale=1)

            gr.Markdown("**常見範例問題（點選後按發送）**")
            with gr.Tabs():
                with gr.Tab("中文"):
                    for group, questions in SAMPLE_GROUPS_ZH.items():
                        gr.Examples(examples=questions, inputs=user_box, label=group)
                with gr.Tab("English"):
                    for group, questions in SAMPLE_GROUPS.items():
                        gr.Examples(examples=questions, inputs=user_box, label=group)

        with gr.Column(scale=2):
            trace_panel = gr.Markdown(EMPTY_TRACE)

    send_button.click(add_user_message, [user_box, chatbot], [user_box, chatbot]).then(
        answer_question, chatbot, [chatbot, trace_panel]
    )
    user_box.submit(add_user_message, [user_box, chatbot], [user_box, chatbot]).then(
        answer_question, chatbot, [chatbot, trace_panel]
    )

    clear_button.click(start_chat, outputs=chatbot).then(
        lambda: ("", EMPTY_TRACE), outputs=[user_box, trace_panel]
    )

if __name__ == "__main__":
    print("正在啟動 Gradio 服務...")
    demo.launch(inbrowser=True) 