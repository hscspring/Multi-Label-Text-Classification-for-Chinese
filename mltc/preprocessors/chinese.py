import os
import re
from pnlp import ptxt, piop
import jieba
import jieba.posseg as pseg


CN_NUM = {
    '〇': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9, '零': 0,
    '壹': 1, '贰': 2, '叁': 3, '肆': 4, '伍': 5,
    '陆': 6, '柒': 7, '捌': 8, '玖': 9, '貮': 2, '两': 2,
}

CN_UNIT = {
    '十': 10,
    '拾': 10,
    '百': 100,
    '佰': 100,
    '千': 1000,
    '仟': 1000,
    '万': 10000,
    '萬': 10000,
    '亿': 100000000,
    '億': 100000000,
    '兆': 1000000000000,
}


class ChineseProcessor:

    """

    Nodes
    ------
    replace info:
    - S ==> Small
    - T ==> Tiny
    - B ==> Big
    - H ==> Huge
    - V ==> Very xx

    - P ==> Person
    - L ==> Location
    - O ==> Organization

    - E ==> English
    """

    def __init__(self):
        self.stopwords_set = set()
        self.pun_zh = r"，。；、？！：“”‘’（）「」『』〔〕【】《》〈〉…——\-—～~·"
        self.pun_en = r",.;?!\(\)\[\]\{\}<>_"
        self.cn_num = "".join(list(CN_NUM.keys()))
        self.cn_unit = "".join(list(CN_UNIT.keys()))
        self.year = "〇一二三四五六七八九十零"
        self.month = "一二三四五六七八九十"
        self.weight = "一二三四五六七八九十百千万亿"

    def reset(self, stopwords_path):
        if stopwords_path and os.path.exists(stopwords_path):
            self.stopwords_set = set(piop.read_lines(stopwords_path))

    def cnnum2num(self, text: str, unit: str):
        rule = re.compile(rf'[{self.cn_num + self.cn_unit}]+{unit}')
        ca = Chinese2Arabic()
        text = rule.sub(lambda x: str(ca(x.group()[:-len(unit)])) + unit, text)
        return text

    def concentration_convert(self, concern: float):
        num = concern
        try:
            num = float(num)
        except Exception as e:
            return concern

        if num < 1.0:
            return "S"
        elif num < 10.0:
            return "T"
        elif num < 100.0:
            return "B"
        elif num < 1000.0:
            return "H"
        else:
            return "V"

    def quantity_convert(self, input_quantity):
        dct = {
            "万": 10000,
            "亿": 100000000
        }
        times = []
        quantity = input_quantity
        while quantity and quantity[-1] in ["万", "亿"]:
            times.append(quantity[-1])
            quantity = quantity[:-1]
        try:
            quantity = float(quantity)
        except Exception as e:
            return input_quantity
        for t in times:
            quantity *= dct.get(t)

        if quantity < 100:
            return "数十"
        elif quantity < 1000:
            return "数百"
        elif quantity < 10000:
            return "数千"
        elif quantity < 100000:
            return "数万"
        elif quantity < 1000000:
            return "数十万"
        elif quantity < 10000000:
            return "数百万"
        elif quantity < 100000000:
            return "数千万"
        elif quantity < 1000000000:
            return "数亿"
        elif quantity < 10000000000:
            return "数十亿"
        elif quantity < 100000000000:
            return "数百亿"
        elif quantity < 1000000000000:
            return "数千亿"
        else:
            return "数"

    def clean_punctuation(self, text):
        rule = re.compile(rf'[{self.pun_zh + self.pun_en}]+')
        text = rule.sub(" ", text)
        return text

    def clean_linkpic(self, text):
        text = ptxt.Text(text, 'pic').clean
        text = ptxt.Text(text, 'lnk').clean
        return text

    def clean_date(self, text):
        rule = re.compile(rf'[\d{self.year}]+年')
        text = rule.sub("X年", text)

        rule = re.compile(rf'[\d{self.month}]+月')
        text = rule.sub("X月", text)

        rule = re.compile(rf'[\d{self.month}]+日')
        text = rule.sub("X日", text)

        # e.g. 11/12/19, 11-1-19, 1.12.19, 11/12/2019
        rule = re.compile(
            r'(?:19|20)\d{2}(?:\/|\-|\.)\d{1,2}(?:\/|\-|\.)\d{1,2}')
        text = rule.sub("X年X月X日", text)
        return text

    def clean_time(self, text):
        rule = re.compile(rf'[\d{self.month}]+[时点]')
        text = rule.sub("X时", text)

        rule = re.compile(rf'[时点][\d{self.month}]+[分]')
        text = rule.sub("时X分", text)
        # e.g. UTC+09:00
        rule = re.compile(r'\w{3}[+-][0-9]{1,2}\:[0-9]{2}\b')
        text = rule.sub("X点", text)
        # e.g. 18:09:01
        rule = re.compile(r'\d{1,2}\:\d{2}\:\d{2}')
        text = rule.sub("X点X分", text)
        # e.g. 18:09
        rule = re.compile(r'\d{1,2}\:\d{2}')
        text = rule.sub("X点X分", text)
        return text

    def clean_money(self, text):
        text = self.cnnum2num(text, "元")

        rule = re.compile(r'\d+[.]?\d*[万亿]?元')
        text = rule.sub(lambda x: self.quantity_convert(
            x.group()[:-1]) + "元", text)
        return text

    def clean_weight(self, text):
        text = self.cnnum2num(text, "千克")
        text = self.cnnum2num(text, "公斤")

        rule = re.compile(rf'[\d{self.weight}]+(?:千克|公斤)')
        text = rule.sub(lambda x: self.quantity_convert(
            x.group()[:-2]) + "千克", text)

        text = self.cnnum2num(text, "斤")
        rule = re.compile(rf'[\d{self.weight}]+斤')
        text = rule.sub(lambda x: self.quantity_convert(
            x.group()[:-1]) + "斤", text)

        text = self.cnnum2num(text, "吨")
        rule = re.compile(rf'[\d{self.weight}]+吨')
        text = rule.sub(lambda x: self.quantity_convert(
            x.group()[:-1]) + "吨", text)

        text = self.cnnum2num(text, "克")
        rule = re.compile(rf'[\d{self.weight}]+克')
        text = rule.sub(lambda x: self.quantity_convert(
            x.group()[:-1]) + "克", text)
        return text

    def clean_concentration(self, text):

        def convert_combine(text):
            pt = ptxt.Text(text, "num")
            pure_text = pt.clean
            converted = self.concentration_convert(pt.extract.mats[0])
            return pure_text + converted

        rule = re.compile(r'浓度\w*[.]?\d+[.]?\d*')
        text = rule.sub(lambda x: convert_combine(x.group()), text)
        return text

    def clean_entity(self, text):
        wps = pseg.cut(text)
        res = []
        for w, pos in wps:
            # 人名
            if pos == "nr":
                res.append("P")
            # 地名
            elif pos == "ns":
                res.append("L")
            # 机构名
            elif pos == "nt":
                res.append("O")
            else:
                res.append(w)
        return "".join(res)

    def clean_stopwords(self, token_list):
        if self.stopwords_set:
            res = [t for t in token_list if t not in self.stopwords_set]
        else:
            res = token_list
        return " ".join(res)

    def clean_nums(self, text):
        rule = re.compile(r"[.-]?[\d.]+[e%]?[\d]?")
        text = rule.sub("X", text)
        return text

    def clean_english(self, text):
        rule = re.compile(r'[a-zA-Z]+')
        text = rule.sub("E", text)
        return text


class ChineseCharProcessor(ChineseProcessor):

    def __init__(self, stopwords_path="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset(stopwords_path)

    def __call__(self, sent):
        sent = ptxt.Text(sent, "whi").clean
        sent = self.clean_linkpic(sent)

        sent = self.clean_english(sent)

        sent = self.clean_date(sent)
        sent = self.clean_time(sent)

        sent = self.clean_money(sent)
        sent = self.clean_weight(sent)
        sent = self.clean_concentration(sent)

        sent = self.clean_entity(sent)

        sent = self.clean_nums(sent)

        clist = list(sent)
        sent = self.clean_stopwords(clist)
        sent = self.clean_punctuation(sent)

        return sent


class ChineseWordProcessor(ChineseProcessor):

    def __init__(self, stopwords_path="", userdict_path="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if userdict_path and os.path.exists(userdict_path):
            jieba.load_userdict(str(userdict_path))
        self.reset(stopwords_path)

    def __call__(self, sent):
        sent = ptxt.Text(sent, "whi").clean
        sent = self.clean_linkpic(sent)

        sent = self.clean_english(sent)

        sent = self.clean_date(sent)
        sent = self.clean_time(sent)

        sent = self.clean_money(sent)
        sent = self.clean_weight(sent)
        sent = self.clean_concentration(sent)

        sent = self.clean_entity(sent)

        sent = self.clean_nums(sent)

        wlist = jieba.lcut(sent)
        sent = self.clean_stopwords(wlist)
        sent = self.clean_punctuation(sent)

        return sent


class Chinese2Arabic:
    """
    Chinese_to_arabic
    modifed from https://github.com/bamtercelboo/corpus_process_script/blob/master/cn_to_arabic/cn_to_arabic.py
    """

    def __init__(self):
        self.CN_NUM = CN_NUM
        self.CN_UNIT = CN_UNIT

    def __call__(self, cn: str):
        unit = 0
        ldig = []
        for cndig in reversed(cn):
            if cndig in self.CN_UNIT:
                unit = self.CN_UNIT.get(cndig)
                if unit == 10000 or unit == 100000000:
                    ldig.append(unit)
                    unit = 1
            else:
                dig = self.CN_NUM.get(cndig)
                if unit:
                    dig *= unit
                    unit = 0
                ldig.append(dig)
        if unit == 10:
            ldig.append(10)
        val, tmp = 0, 0
        for x in reversed(ldig):
            if x == 10000 or x == 100000000:
                val += tmp * x
                tmp = 0
            else:
                tmp += x
        val += tmp
        if val == 0 and cn != "零":
            return cn
        else:
            return val


if __name__ == '__main__':
    ccp = ChineseCharProcessor(stopwords_path="../dicts/stopwords.txt")
    cwp = ChineseWordProcessor(stopwords_path="../dicts/stopwords.txt")
    text = """
    一元，三里，十元。
    朱镕基总理不错。张三去爬珠穆朗玛峰。
    多福多寿一千万元啊，这是两百元。给你。我与你，也好。19.42万元，共8万元。18.32,万，千，百，亿元。
    123千克，三百二十千克，两百多千克，一百二十公斤，1万千克，20000千克，好多。
    3043克白粉，20斤白面，3000吨钢材，三千吨钢材。
    浓度达214,浓度分别超国家规定的排放标准8.38
    """

    res = ccp(text)
    print(res)

    res = cwp(text)
    print(res)

    print(text)

    print(ccp.clean_money(text))
    print(ccp.clean_weight(text))
