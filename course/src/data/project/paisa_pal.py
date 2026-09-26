# paisa_pal.py: the shared data for the Paisa Pal support-bot project (Parts 8 to 10).
# Every project exercise starts with this file already loaded. Small on purpose: it all runs in a browser.
import numpy as np

# ---- the help centre: what the bot is allowed to know ----
HELP_PAGES = {
    "refunds": "If a UPI payment fails but money is deducted, the refund reaches your bank account within 3 working days. If it has not arrived after 3 working days, raise a dispute from the transaction screen.",
    "cashback": "Cashback is credited within 48 hours of an eligible payment. There is no cashback on wallet top-ups or on payments below 100 rupees.",
    "kyc": "Full KYC needs a PAN card and an Aadhaar-linked mobile number. Without full KYC the wallet limit is 10,000 rupees a month.",
    "security": "After five wrong PIN attempts the account is locked for 30 minutes. Paisa Pal staff will never ask for your PIN or OTP.",
    "limits": "You can send up to 1,00,000 rupees a day by UPI. A new device can send at most 5,000 rupees in its first 24 hours.",
    "disputes": "A dispute is reviewed within 7 working days. You get an SMS and an in-app message when it is resolved.",
}

# ---- real questions customers asked, with the page that answers them and a fact the answer must contain ----
# (question, page id or None when no page answers it, required fact or None)
QUESTIONS = [
    ("my upi payment failed but the money was deducted, when do i get it back", "refunds", "3 working days"),
    ("i paid 50 rupees, why no cashback", "cashback", "100 rupees"),
    ("how long does cashback take to show up", "cashback", "48 hours"),
    ("what documents do i need for full kyc", "kyc", "PAN"),
    ("why is my wallet limit only 10000", "kyc", "10,000"),
    ("my account is locked after wrong pin", "security", "30 minutes"),
    ("someone from paisa pal called asking for my otp", "security", "never ask"),
    ("how much can i send in a day", "limits", "1,00,000"),
    ("new phone, cannot send 20000 rupees", "limits", "5,000"),
    ("how long does a dispute take", "disputes", "7 working days"),
    ("can you recommend a good biryani place in bengaluru", None, None),
    ("what is the share price of paisa pal", None, None),
]

# ---- the bot's tools: what it can do, not just say ----
CUSTOMERS = {
    "C101": {"name": "Riya", "balance": 2450.0, "kyc": "full"},
    "C102": {"name": "Dev", "balance": 180.5, "kyc": "minimum"},
}
TRANSACTIONS = {
    "T9001": {"customer": "C101", "amount": 1200.0, "status": "failed", "refund": "in progress"},
    "T9002": {"customer": "C102", "amount": 50.0, "status": "success", "refund": None},
}
TOOLS = {
    "get_balance": {"args": {"customer_id": "str"}, "moves_money": False},
    "refund_status": {"args": {"transaction_id": "str"}, "moves_money": False},
    "send_money": {"args": {"customer_id": "str", "to": "str", "amount": "float"}, "moves_money": True},
}

def get_balance(customer_id):
    return CUSTOMERS[customer_id]["balance"]

def refund_status(transaction_id):
    return TRANSACTIONS[transaction_id]["refund"] or "no refund for this transaction"

# ---- helpers every exercise may use ----
def tokenize(text):
    """Lower-case words and numbers; punctuation dropped. Good enough for a toy bot."""
    out, word = [], ""
    for ch in text.lower():
        if ch.isalnum():
            word += ch
        elif word:
            out.append(word); word = ""
    if word:
        out.append(word)
    return out

def softmax(z, axis=-1):
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)
