# coding=utf8

import re

# Common patterns
re_latin_mwe = re.compile(ur"^[-a-z ]+$", re.U|re.I)
re_latin_word = re.compile(ur"^[-a-z]+$", re.U|re.I)
re_latin_chars = re.compile(r"[a-zA-Z]")
re_whitespaces = re.compile(r"\s+")
re_newlines = re.compile(r"[\n\r]+")
re_xml_escape = re.compile(r"&[^\s]+;")
re_dollar = re.compile(r"\$")
re_phone = re.compile(r"\+?\s*\b[78]\s*[\(\s-]?\d\d\d[)\s-]?\s*(\d[-\s]?){7}\b")
re_url = re.compile(ur"https?://[^ ]+", re.U|re.I)
re_exclamation = re.compile(ur"!!!+", re.U|re.I)
re_question = re.compile(ur"\?\?\?+", re.U|re.I)
re_upper_en = re.compile(ur"[A-Z]",re.U)
re_upper_en_words = re.compile(ur"[A-Z]{3,}",re.U)
re_spaced_numbers = re.compile(ur"^[,\d\s]+$", re.U|re.I)
re_number = re.compile(ur"\d+", re.U|re.I)
re_numeric_stops = re.compile(ur"[\d\s:;\.-]+(?![а-яa-z\d-])", re.U|re.I)
re_at = re.compile(ur"@", re.U|re.I)

re_amp = re.compile(r"&(amp|quot)(\\;|;)?", re.I)
re_escape = re.compile(r"\\;", re.I)
re_quote_escape = re.compile(r'[\\"]', re.I)

# Sentiment patterns
re_pos_simley = re.compile(r"[:;8%=][-=]?[\)\]]+", re.I)
re_pos_simley2 = re.compile(r"\s+\){2,}", re.I)
re_neg_smiley = re.compile(r"[:;8%=][-=]?[\(\[]+", re.I)
re_neg_simley2 = re.compile(r"\s+\({2,}", re.I)


re_norm_babel = re.compile(ur"[()_:]", re.U|re.I)
re_norm_babel_dash = re.compile(ur"[()_:-]", re.U|re.I)
re_whitespaces2 = re.compile(r"\s+")