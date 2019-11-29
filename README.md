# README

This is a repo focused on Multi-Label Text Classification, the main structure was forked from [lonePatient/Bert-Multi-Label-Text-Classification](https://github.com/lonePatient/Bert-Multi-Label-Text-Classification). We did several improvements:

- Add a pipeline to automatically configure the whole thing
- Add a preprocessor for Chinese
- Add an engineering part
- Add a basic tokenizer

## Environment

```bash
# create a venv and install the dependencies
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

- Prepare dataset

    - Can be download here: [dataset](https://pan.baidu.com/s/1evLbl4Iyl94khO03aQwaWQ), `8vkf`

    - Add `train.csv` and `test.csv` to `dataset/`

    - Each line of the `train.csv` has two fields (fact and meta), like:

        ```python
        "fact": "2015年11月5日上午，被告人胡某在平湖市乍浦镇的嘉兴市多凌金牛制衣有限公司车间内，与被害人孙某因工作琐事发生口角，后被告人胡某用木制坐垫打伤被害人孙某左腹部。经平湖公安司法鉴定中心鉴定：孙某的左腹部损伤已达重伤二级。",   
        "meta": 
        {  
            "relevant_articles": [234],  
            "accusation": ["故意伤害"], 
            "criminals": ["胡某"],  
            "term_of_imprisonment": 
            {  
                "death_penalty": false,  
                "imprisonment": 12,  
                "life_imprisonment": false
            }
        }
        ```

    - Each line of the `test.csv` has only one field: `fact`

    - Paths and filenames can be defined in `configs/basic_config.py`

- Prepare pretrained model

    - Add pretrained files to `pretrain/bert/base-uncased/`, here we used a [domain model](https://github.com/thunlp/OpenCLaP)
    - Paths and filenames can be defined in `configs/basic_config.py`
    - For example bert filenames should be: `config.json`, `pytorch_model.bin` and `bert_vocab.txt`

- Define a pipeline

    - Edit `pipeline.yml` 
    - We have already added several pipelines in the file
    - When the pipeline has been changed, it's better to clean the cached dataset

- Run `./run.sh`

    - Also could run by hand

    ```bash
    python main.py --do_data
    python main.py --do_train --save_best
    python main.py --do_test
    ```

    - Or set the train, test data number

    ```bash
    python main.py --do_data --train_data_num 100
    python main.py --do_train --save_best
    python main.py --do_test --test_data_num 10
    ```

## Dataset

[thunlp/CAIL: Chinese AI & Law Challenge](https://github.com/thunlp/CAIL) Task 1

## Models

- TextCNN
- Bert basic model
- Bert + TextCNN
- Bert + TextRCNN
- Bert + TextDPCNN

## Changelog

- 191127 updated usage details

- 191126 created