import os
import sys
import pytest

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

from chinese import ChineseProcessor, Chinese2Arabic

@pytest.fixture
def setup():
    cp = ChineseProcessor()
    return cp


def test_clean_nums(setup):
    text = "1.3的-3说法 2% 发生2000的 1e9 发 1发"
    new = setup.clean_nums(text)
    assert new == "X的X说法 X 发生X的 X 发 X发"


def test_clean_punctuation(setup):
    text = ",a.?!哈哈？，。我！你……《》就是~不”“说：嘿嘿"
    new = setup.clean_punctuation(text)
    assert new.split() == "a 哈哈 我 你 就是 不 说 嘿嘿".split()


def test_clean_linkpic(setup):
    inp = """https://www.yam.com 或 
    [网址](https://www.google.cn/)和
    ![](http://xx.jpg)"""
    new = setup.clean_linkpic(inp)
    assert new.replace("\n", "").replace(" ", "") == "或和"


def test_cnnum2num(setup):
    text = "一万五千六百三十八元"
    new = setup.cnnum2num(text, "元")
    assert new == "15638元"

    text = "一百二十公斤"
    new = setup.cnnum2num(text, "公斤")
    assert new == "120公斤"

    text = "300吨"
    new = setup.cnnum2num(text, "吨")
    assert new == "300吨"

    text = "1万克"
    new = setup.cnnum2num(text, "克")
    assert new == "1万克"


def test_concentration_convert(setup):
    inp = 2
    new = setup.concentration_convert(inp)
    assert new == "T"

    inp = 2000.32342
    new = setup.concentration_convert(inp)
    assert new == "V"

    inp = "哈哈"
    new = setup.concentration_convert(inp)
    assert new == "哈哈"


def test_quantity_convert(setup):
    inp = "2万"
    new = setup.quantity_convert(inp)
    assert new == "数万"

    inp = "3000"
    new = setup.quantity_convert(inp)
    assert new == "数千"

    inp = "500.232"
    new = setup.quantity_convert(inp)
    assert new == "数百"

    inp = "3亿"
    new = setup.quantity_convert(inp)
    assert new == "数亿"

    inp = "哈哈"
    new = setup.quantity_convert(inp)
    assert new == "哈哈"

    inp = "3万多"
    new = setup.quantity_convert(inp)
    assert new == "3万多"

    inp = "30多万"
    new = setup.quantity_convert(inp)
    assert new == "30多万"

    inp = "3万亿"
    new = setup.quantity_convert(inp)
    assert new == "数"


def test_clean_punctuation(setup):
    inp = "我，你。他？哈,后~《爱》"
    new = setup.clean_punctuation(inp)
    assert new == "我 你 他 哈 后 爱 "


def test_clean_date(setup):
    inp = "2018年，18年3月，3月2日，2日晚，2018-11-1，1987/3/25，1987.04.22"
    new = setup.clean_date(inp)
    assert new == "X年，X年X月，X月X日，X日晚，X年X月X日，X年X月X日，X年X月X日"


def test_clean_time(setup):
    inp = "十点，十点三十分，八时整，UTC+09:00，18:09:01，18:09"
    new = setup.clean_time(inp)
    assert new == "X时，X时X分，X时整，X点，X点X分，X点X分"


def test_clean_money(setup):
    inp = "十元，三里，一千万元啊，这是两百元。给你。19.42万元，共8万元。18.32,万，千，百，亿元。"
    new = setup.clean_money(inp)
    assert new == "数十元，三里，数千万元啊，这是数百元。给你。数十万元，共数万元。18.32,万，千，百，亿元。"


def test_clean_weight(setup):
    inp = "123千克，三百二十克，两百多吨，一百二十公斤，1万斤，20000吨，好多。"
    new = setup.clean_weight(inp)
    assert new == "数百千克，数百克，两百多吨，数百千克，数万斤，数万吨，好多。"

    inp = "3043克白粉，20斤白面，3000吨钢材，三千吨钢材。"
    new = setup.clean_weight(inp)
    assert new == "数千克白粉，数十斤白面，数千吨钢材，数千吨钢材。"


def test_clean_concentration(setup):
    inp = "浓度达214,浓度分别超国家规定的排放标准8.38"
    new = setup.clean_concentration(inp)
    assert new == "浓度达H,浓度分别超国家规定的排放标准T"


def test_clean_entity(setup):
    inp = "永顺县人民检察院指控，张三去爬珠穆朗玛峰了。"
    new = setup.clean_entity(inp)
    assert new == "LO指控，P去爬L了。"


def test_clean_stopwords(setup):
    inp = ["我", "喜欢", "你"]
    new = setup.clean_stopwords(inp)
    assert new == " ".join(inp)

    root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    stopwords_path = os.path.join(root, "dicts", "stopwords.txt")
    assert os.path.exists(stopwords_path) == True
    setup.reset(stopwords_path)

    inp = ["我", "喜欢", "你"]
    new = setup.clean_stopwords(inp)
    assert new == "喜欢"


def test_clean_english(setup):
    inp = "Lenovo 是联想，Alibaba 是阿里巴巴。"
    new = setup.clean_english(inp)
    assert new == "E 是联想，E 是阿里巴巴。"


def test_chinese2arabic():
    ca = Chinese2Arabic()
    s = "一亿三千万"
    assert ca(s) == 130000000
    s = "一万五千六百三十八"
    assert ca(s) == 15638
    s = "壹仟两百"
    assert ca(s) == 1200
    s = "十一"
    assert ca(s) == 11
    s = "三"
    assert ca(s) == 3
    s = "两百五十"
    assert ca(s) == 250
    s = "两百零五"
    assert ca(s) == 205
    s = "二十万五千"
    assert ca(s) == 205000
    s = "两百三十九万四千八百二十三"
    assert ca(s) == 2394823
    s = "一千三百万"
    assert ca(s) == 13000000
    s = "万"
    assert ca(s) == "万"
    s = "亿"
    assert ca(s) == "亿"
    s = "千"
    assert ca(s) == "千"
    s = "百"
    assert ca(s) == "百"
    s = "零"
    assert ca(s) == 0


if __name__ == '__main__':
    print(root)
