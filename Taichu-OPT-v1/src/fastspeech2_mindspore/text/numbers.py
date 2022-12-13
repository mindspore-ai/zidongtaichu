# from https://github.com/keithito/tacotron 

import re
import inflect


_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")


def _remove_commas(m):
    """remove_commas"""
    return m.group(1).replace(",", "")


def _expand_decimal_point(m):
    """expand_decimal_point"""
    return m.group(1).replace(".", " point ")


def _expand_dollars(m):
    """expand_dollars"""
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    if dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    if cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    return "zero dollars"


def _expand_ordinal(m):
    """expand_ordinal"""
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    """expand_number"""
    num = int(m.group(0))
    if 1000 < num < 3000:
        if num == 2000:
            expand_num = "two thousand"
        elif 2000 < num < 2010:
            expand_num = "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            expand_num = _inflect.number_to_words(num // 100) + " hundred"
        else:
            expand_num = _inflect.number_to_words(
                num, andword="", zero="oh", group=2
            ).replace(", ", " ")
    else:
        expand_num = _inflect.number_to_words(num, andword="")
    return expand_num


def normalize_numbers(text):
    """normalize_numbers"""
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text
