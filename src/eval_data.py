"""人工标注测试集(Day4)。

每题:
  question  - 问题
  type      - "in"(库内,应作答) / "out"(库外,应拒答)
  gold_docs - 库内题:答案"应该来自"哪些文档(doc stem)。topic 有重叠时给多个。
              用于算检索命中率:检索回来的块里只要有 gold_docs 中的文档即算命中。

gold 是"我作为标注者凭领域知识"独立判定的权威来源,不是看检索结果倒推的。
"""

# 文档 stem:bis_aml_cft_guidelines / eu_gdpr / Fees-and-charges-first-abu-dhabi-bank
#          / GeneralTermsAndConditions / fab-consolidated-credit-cards

_FAB_CARD = "fab-consolidated-credit-cards"
_FAB_FEES = "Fees-and-charges-first-abu-dhabi-bank"
_ENBD_GTC = "GeneralTermsAndConditions"
_BIS = "bis_aml_cft_guidelines"
_GDPR = "eu_gdpr"

TESTSET = [
    # --- AML / BIS (5) ---
    {"question": "What should a bank's customer acceptance policy include?",
     "type": "in", "gold_docs": [_BIS]},
    {"question": "What enhanced due diligence measures apply to higher-risk customers?",
     "type": "in", "gold_docs": [_BIS]},
    {"question": "What are the three lines of defence in money-laundering risk management?",
     "type": "in", "gold_docs": [_BIS]},
    {"question": "How should a bank identify and verify the beneficial owner of an account?",
     "type": "in", "gold_docs": [_BIS]},
    {"question": "What ongoing monitoring of customer transactions does the AML guidance require?",
     "type": "in", "gold_docs": [_BIS]},

    # --- GDPR (5) ---
    {"question": "What rights do data subjects have under the GDPR?",
     "type": "in", "gold_docs": [_GDPR]},
    {"question": "Within what time must a personal data breach be notified to the supervisory authority?",
     "type": "in", "gold_docs": [_GDPR]},
    {"question": "When is consent a valid lawful basis for processing personal data?",
     "type": "in", "gold_docs": [_GDPR]},
    {"question": "What is the right to erasure (right to be forgotten)?",
     "type": "in", "gold_docs": [_GDPR]},
    {"question": "When must an organisation designate a Data Protection Officer?",
     "type": "in", "gold_docs": [_GDPR]},

    # --- FAB credit cards / fees (4) ---
    {"question": "What is the cash advance fee on FAB credit cards?",
     "type": "in", "gold_docs": [_FAB_CARD, _FAB_FEES]},
    {"question": "How long is the interest-free period on FAB credit card purchases?",
     "type": "in", "gold_docs": [_FAB_CARD]},
    {"question": "What foreign currency transaction fee applies to FAB credit cards?",
     "type": "in", "gold_docs": [_FAB_CARD, _FAB_FEES]},
    {"question": "What fee does FAB charge for closing a personal account?",
     "type": "in", "gold_docs": [_FAB_FEES]},

    # --- Emirates NBD general T&Cs (1) ---
    {"question": "What do the bank's general terms and conditions say about dormant accounts?",
     "type": "in", "gold_docs": [_ENBD_GTC, _FAB_FEES]},

    # --- 库外:应拒答 (5) ---
    {"question": "What is the interest rate on a Tesla car loan in Japan?", "type": "out", "gold_docs": []},
    {"question": "Who won the 2022 FIFA World Cup?", "type": "out", "gold_docs": []},
    {"question": "How do I bake a chocolate cake?", "type": "out", "gold_docs": []},
    {"question": "What is the capital of France?", "type": "out", "gold_docs": []},
    {"question": "What are the Bitcoin mining rewards in 2025?", "type": "out", "gold_docs": []},
]
