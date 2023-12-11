# Transformer

**Transformer: Vanilla Transformer for Time Series Forecasting**

![]('./img/transformer.png)

## Usage

1. Install modules following command.

```
pip install -r requirements.txt
```

2. Prepare Dataset. Download the [ETT Dataset](https://github.com/zhouhaoyi/ETDataset) and place the downloaded data in `./data` directory.

3. Train model.

```
python -m src.train
```

## Reference

- [Paper] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Repo] [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [Blog] [Time Series Transformers - Hugging Face](https://huggingface.co/docs/transformers/model_doc/time_series_transformer)