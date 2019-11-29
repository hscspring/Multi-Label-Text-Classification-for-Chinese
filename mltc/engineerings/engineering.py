from collections import Counter
import pandas as pd
import jieba
import jieba.analyse

from pnlp import ptxt

ALLOW_POS = (
    'a', 'ad', 'ag', 'an',
    'n', 'ng', 'nr', 'nrfg', 'nrt', 'ns', 'nt', 'nz',
    'vn', 'v'
)

class Engineer:

    def __init__(self, text_list: list):
        self.df = pd.DataFrame(text_list, columns=["text"])

    @property
    def length_related_features(self) -> pd.DataFrame:
        data = self.df.copy()

        data["len"] = data["text"].apply(len)

        need_cols = [
            "len"
        ]

        return data[need_cols]

    @property
    def word_related_features(self) -> pd.DataFrame:
        data = self.df.copy()

        pzh = re.compile(r'[\u4e00-\u9fa5]+')

        data["words"] = data["text"].apply(
            lambda x: jieba.lcut(x))
        data["words_zh"] = data["words"].apply(
            lambda x: [w for w in x if pzh.search(w)])

        data["word_num"] = data["words"].apply(lambda x: len(x))

        data["punc_num"] = data["text"].apply(lambda x: ptxt.Text(x).len_pun)
        data["punc_num_ratio"] = data["punc_num"] / data["word_num"]

        data["num_num"] = data["text"].apply(lambda x: ptxt.Text(x).len_num)
        data["num_num_ratio"] = data["num_num"] / data["word_num"]

        data["appear_once"] = data["words_zh"].apply(
            lambda x: len([w for (w, f) in Counter(x).items() if f == 1]))
        data["hapax_legomena1"] = data["appear_once"] / data["word_num"]

        data["appear_twice"] = data["words_zh"].apply(
            lambda x: len([w for (w, f) in Counter(x).items() if f == 2]))
        data["hapax_legomena2"] = data["appear_twice"] / data["word_num"]

        data["one_char_num"] = data["words_zh"].apply(
            lambda x: len([w for w in x if len(w) == 1]))
        data["one_char_ratio"] = data["one_char_num"] / data["word_num"]

        data["two_char_num"] = data["words_zh"].apply(
            lambda x: len([w for w in x if len(w) == 2]))
        data["two_char_ratio"] = data["two_char_num"] / data["word_num"]

        data["three_char_num"] = data["words_zh"].apply(
            lambda x: len([w for w in x if len(w) == 3]))
        data["three_char_ratio"] = data["three_char_num"] / data["word_num"]

        data["four_char_num"] = data["words_zh"].apply(
            lambda x: len([w for w in x if len(w) == 4]))
        data["four_char_ratio"] = data["four_char_num"] / data["word_num"]

        data["ttr"] = data["words"].apply(
            lambda x: len(set(x)) / len(x))

        need_cols = [
            "word_num",
            "punc_num_ratio",
            "num_num_ratio",
            "hapax_legomena1",
            "hapax_legomena2",
            "one_char_ratio",
            "two_char_ratio",
            "three_char_ratio",
            "four_char_ratio",
            "ttr"
        ]
        return data[need_cols]

    @property
    def sent_related_features(self) -> pd.DataFrame:
        data = self.df.copy()

        rule = re.compile(r'[，、。？！”……]+')
        data["short_sent_num"] = data["text"].apply(
            lambda x: len([ss for ss in rule.split(x) if len(ss) > 1]))

        rule = re.compile(r'[。？！”……]+')
        data["sent_num"] = data["text"].apply(
            lambda x: len([ss for ss in rule.split(x) if len(ss) > 1]))

        need_cols = [
            "short_sent_num",
            "sent_num"
        ]

        return data[need_cols]

    @property
    def content_related_features(self) -> pd.DataFrame:
        data = self.df.copy()

        # 被告数量
        rule = re.compile(r'被告')
        data["defendant_num"] = data["text"].apply(
            lambda x: len(rule.findall(x)))

        # 被告中男性比例
        rule = re.compile(r'被告.*男.*[。！？……”]+')
        data["defendant_male_num"] = data["text"].apply(
            lambda x: len(rule.findall(x)))
        data["defendante_male_ratio"] = data[
            "defendant_male_num"] / data["defendant_num"]

        # 被告中法定代表人比例
        rule = re.compile(r'被告.*法定代表人.*[。！？……”]+')
        data["defendant_company_num"] = data["text"].apply(
            lambda x: len(rule.findall(x)))
        data["defendante_company_ratio"] = data[
            "defendant_company_num"] / data["defendant_num"]

        # 担保
        rule = re.compile(r'担保')
        data["guarantee_num"] = data["text"].apply(
            lambda x: len(rule.findall(x)))

        need_cols = [
            "defendant_num",
            "defendante_male_ratio",
            "defendante_company_ratio",
            "guarantee_num"
        ]

        return data[need_cols]

    @property
    def keywords(self):
        data = self.df.copy()

        data["keywords"] = data["text"].apply(
            lambda x: KeyWords(x)(10))

        return data[["keywords"]]

    def __call__(self):
        data = pd.concat([
            self.length_related_features,
            self.word_related_features,
            self.sent_related_features,
            self.content_related_features
            ], axis=1)
        # qt = QuantileTransformer(random_state=2019)
        # qt_features = qt.fit_transform(data)
        return data


class KeyWords:
    
    def __init__(self, text):
        self.text = text

    @property
    def tfidf(self) -> list:
        kw_with_weight = jieba.analyse.extract_tags(
            self.text, allowPOS=ALLOW_POS, withWeight=True)
        return self.standardize(kw_with_weight)

    @property
    def textrank(self) -> list:
        kw_with_weight = jieba.analyse.textrank(
            self.text, allowPOS=ALLOW_POS, withWeight=True)
        return self.standardize(kw_with_weight)

    def standardize(self, kw_with_weight: list) -> list:
        words, weights = [], []
        for w, p in kw_with_weight:
            words.append(w)
            weights.append(p)
        arr = np.array(weights)
        sumw = np.sum(arr)
        new_weights = arr / sumw
        kw_standardized = [(words[i], new_weights[i])
                           for i in range(len(words))]
        return kw_standardized

    def __call__(self, topk=5) -> list:
        union_kwd = {}
        idf_kwd = dict(self.tfidf)
        trk_kwd = dict(self.textrank)
        union = set(idf_kwd.keys()) & set(trk_kwd.keys())
        for w in union:
            union_kwd[w] = idf_kwd[w] + trk_kwd[w]
        sort_kws = sorted(union_kwd.items(), key=lambda x: x[1], reverse=True)
        res = [w for (w,f) in std_kws][:topk]
        return res


