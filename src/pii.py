"""输入侧 PII 检测 + 脱敏(Day3)。

引擎:Microsoft Presidio(内置卡号/邮箱/电话/IBAN/人名等识别器)
     + 自定义 Emirates ID 识别器(Presidio 没有的本地化 PII)。
处理:脱敏后继续——把 PII 替换成 <ENTITY_TYPE> 占位符,问题照常回答。

为什么放在最入口:用户问题会流向日志、检索、LLM,任一处泄露都是合规事故,
所以在进入管线前就脱敏,后面所有环节只看得到脱敏版。
"""

from __future__ import annotations

import functools

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine


@functools.lru_cache(maxsize=1)
def _get_engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    """构建一次,缓存复用(加载 spaCy 模型很贵)。"""
    # 用 en_core_web_sm(小模型,省磁盘);代价:人名/地点 NER 弱于 lg。
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    })
    analyzer = AnalyzerEngine(
        nlp_engine=provider.create_engine(),
        supported_languages=["en"],
    )

    # —— 自定义识别器:Emirates ID(格式 784-YYYY-NNNNNNN-N,共 15 位)——
    # 这是 Presidio 内置没有的本地化 PII。展示"扩展 Presidio"的能力。
    emirates_id = PatternRecognizer(
        supported_entity="EMIRATES_ID",
        patterns=[Pattern(
            name="emirates_id",
            regex=r"\b784[- ]?\d{4}[- ]?\d{7}[- ]?\d\b",
            score=0.9,
        )],
        context=["emirates", "id", "eid", "identity"],  # 附近出现这些词时提分
    )
    analyzer.registry.add_recognizer(emirates_id)
    # 注:UAE IBAN(AE+21位)由 Presidio 内置 IBAN_CODE 识别器覆盖,无需自定义。

    return analyzer, AnonymizerEngine()


# 只检测银行场景真正敏感的 PII 类型 = 白名单。
# 排除 LOCATION / DATE_TIME / URL / NRP 等 —— 小模型 NER 在这些上误报多
# (例:把 "Emirates" 误判成 LOCATION),且它们不是金融敏感信息。
# 这是 precision/recall 权衡:宁可少抹噪音,保住查询语义。
PII_ENTITIES = [
    "CREDIT_CARD",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IBAN_CODE",
    "PERSON",
    "US_SSN",
    "EMIRATES_ID",   # 自定义
]

# 输出泄露扫描只查"结构化账户类 PII":卡号/邮箱/电话/IBAN/SSN/Emirates ID。
# 故意不含 PERSON —— 合规文本里人名很常见(如法规署名),不算泄露,否则误拦。
LEAK_ENTITIES = [e for e in PII_ENTITIES if e != "PERSON"]


def mask_pii(text: str, entities: list[str] | None = None) -> tuple[str, list[str]]:
    """返回 (脱敏后文本, 命中的 PII 类型列表)。

    AnonymizerEngine 默认把命中片段替换成 <ENTITY_TYPE>,如 <CREDIT_CARD>。
    """
    analyzer, anonymizer = _get_engines()
    results = analyzer.analyze(
        text=text, language="en",
        entities=entities or PII_ENTITIES, score_threshold=0.4,
    )
    masked = anonymizer.anonymize(text=text, analyzer_results=results)
    findings = sorted({r.entity_type for r in results})
    return masked.text, findings


if __name__ == "__main__":
    sample = ("My card is 4111 1111 1111 1111, Emirates ID 784-1990-1234567-1, "
              "email me at john.doe@example.com — what is my late payment fee?")
    masked, found = mask_pii(sample)
    print("原文:", sample)
    print("脱敏:", masked)
    print("命中:", found)
