# -*- coding: utf-8 -*-
import traceback
import re

class TextNormalizer:
    def __init__(self):
        self.zh_normalizer = None
        self.en_normalizer = None
        self.char_rep_map = {
            "：": ",",
            "；": ",",
            ";": ",",
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "\n": " ",
            "·": "-",
            "、": ",",
            "...": "…",
            "……": "…",
            "$": ".",
            "“": "'",
            "”": "'",
            '"': "'",
            "‘": "'",
            "’": "'",
            "（": "'",
            "）": "'",
            "(": "'",
            ")": "'",
            "《": "'",
            "》": "'",
            "【": "'",
            "】": "'",
            "[": "'",
            "]": "'",
            "—": "-",
            "～": "-",
            "~": "-",
            "「": "'",
            "」": "'",
            ":": ",",
        }

    def load(self):
        import platform
        try:
            if platform.system() == "Darwin":
                from wetext import Normalizer
                self.zh_normalizer = Normalizer(lang="zh", operator="tn")
                self.en_normalizer = Normalizer(lang="en", operator="tn")
            else:
                from tn.chinese.normalizer import Normalizer as NormalizerZh
                self.zh_normalizer = NormalizerZh()
                try:
                    from tn.english.normalizer import Normalizer as NormalizerEn
                    self.en_normalizer = NormalizerEn()
                except ImportError:
                    print("⚠️ 未找到 tn.english.normalizer，跳过英文规范化器。")
                    self.en_normalizer = None
        except Exception:
            print("❌ 加载 Normalizer 失败：")
            print(traceback.format_exc())

    def match_email(self, email):
        pattern = r'^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$'
        return re.match(pattern, email) is not None

    PINYIN_TONE_PATTERN = r"([bmnpqdfghjklzcsxwy]?h?[aeiouüv]{1,2}[ng]*|ng)([1-5])"

    def use_chinese(self, s):
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', s))
        has_alpha = bool(re.search(r'[a-zA-Z]', s))
        is_email = self.match_email(s)
        if has_chinese or not has_alpha or is_email:
            return True
        has_pinyin = bool(re.search(self.PINYIN_TONE_PATTERN, s, re.IGNORECASE))
        return has_pinyin

    def infer(self, text: str):
        if not self.zh_normalizer:
            print("❌ zh_normalizer 未初始化")
            return ""
        replaced_text, pinyin_list = self.save_pinyin_tones(text.rstrip())

        try:
            normalizer = self.zh_normalizer if self.use_chinese(replaced_text) else self.en_normalizer
            if normalizer:
                result = normalizer.normalize(replaced_text)
            else:
                result = replaced_text
        except Exception:
            result = ""
            print(traceback.format_exc())

        result = self.restore_pinyin_tones(result, pinyin_list)
        pattern = re.compile("|".join(re.escape(p) for p in self.char_rep_map.keys()))
        result = pattern.sub(lambda x: self.char_rep_map[x.group()], result)
        return result

    def correct_pinyin(self, pinyin):
        if pinyin[0] not in "jqx":
            return pinyin
        pattern = r"([jqx])[uü](n|e|an)*(\d)"
        repl = r"\g<1>v\g<2>\g<3>"
        pinyin = re.sub(pattern, repl, pinyin)
        return pinyin

    def save_pinyin_tones(self, original_text):
        origin_pinyin_pattern = re.compile(self.PINYIN_TONE_PATTERN, re.IGNORECASE)
        original_pinyin_list = re.findall(origin_pinyin_pattern, original_text)
        if len(original_pinyin_list) == 0:
            return (original_text, None)
        original_pinyin_list = list(set(''.join(p) for p in original_pinyin_list))
        transformed_text = original_text
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(pinyin, f"<pinyin_{number}>")
        return transformed_text, original_pinyin_list

    def restore_pinyin_tones(self, normalized_text, original_pinyin_list):
        if not original_pinyin_list or len(original_pinyin_list) == 0:
            return normalized_text
        transformed_text = normalized_text
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            pinyin = self.correct_pinyin(pinyin)
            transformed_text = transformed_text.replace(f"<pinyin_{number}>", pinyin)
        return transformed_text
