"""Tests para los modelos de la Etapa 4 (Baseline, Inferencia y pipeline de BETO)."""

import pandas as pd
import pytest
import torch

from src.models.predict import predict_parquet, predict_texts
from src.models.train_baseline import (
    build_baseline_pipeline,
    load_config,
    save_baseline_artifacts,
    train_and_eval_model,
)
from src.models.train_beto import TextClassificationDataset, get_label_mapping


@pytest.fixture
def mini_dataset():
    """Dataset sintético mínimo balanceado para pruebas rápidas."""
    records = [
        {
            "doc_id": "1",
            "user_id": "u1",
            "text_clean": "hoy me siento muy feliz y con energía",
            "label": 0,
        },
        {
            "doc_id": "2",
            "user_id": "u1",
            "text_clean": "lindo día para salir a pasear con amigos",
            "label": 0,
        },
        {
            "doc_id": "3",
            "user_id": "u2",
            "text_clean": "todo marcha excelente en el trabajo",
            "label": 0,
        },
        {
            "doc_id": "4",
            "user_id": "u3",
            "text_clean": "no tengo ganas de nada, me siento muy triste y vacío",
            "label": 2,
        },
        {
            "doc_id": "5",
            "user_id": "u3",
            "text_clean": "otra noche sin poder dormir con mucha angustia",
            "label": 2,
        },
        {
            "doc_id": "6",
            "user_id": "u4",
            "text_clean": "siempre me pasa lo mismo, todo es un desastre",
            "label": 2,
        },
        {
            "doc_id": "7",
            "user_id": "u5",
            "text_clean": "muy buen fin de semana descansando",
            "label": 0,
        },
        {
            "doc_id": "8",
            "user_id": "u6",
            "text_clean": "estoy destrozado por dentro y no veo salida",
            "label": 2,
        },
    ]
    df = pd.DataFrame(records)
    df_train = df.iloc[:4].copy()
    df_val = df.iloc[4:6].copy()
    df_test = df.iloc[6:8].copy()
    return df_train, df_val, df_test


def test_load_models_config():
    cfg = load_config("configs/models.yaml")
    assert "baseline" in cfg
    assert "beto" in cfg
    assert "tfidf" in cfg["baseline"]
    assert "logreg" in cfg["baseline"]
    assert "svm" in cfg["baseline"]


def test_build_baseline_pipeline_logreg():
    pipe = build_baseline_pipeline("logreg")
    assert "tfidf" in pipe.named_steps
    assert "clf" in pipe.named_steps
    assert hasattr(pipe.named_steps["clf"], "predict_proba")


def test_build_baseline_pipeline_svm():
    pipe = build_baseline_pipeline("svm")
    assert "tfidf" in pipe.named_steps
    assert "clf" in pipe.named_steps
    assert hasattr(pipe.named_steps["clf"], "decision_function")


def test_train_and_eval_baseline_logreg(mini_dataset):
    df_train, df_val, df_test = mini_dataset
    pipe, res = train_and_eval_model("logreg", df_train, df_val, df_test, seed=42)

    assert "val" in res
    assert "test" in res
    assert 0.0 <= res["test"]["accuracy"] <= 1.0
    assert 0.0 <= res["test"]["f1_macro"] <= 1.0
    assert "auc_ovr_macro" in res["test"] or res["test"]["f1_macro"] >= 0.0


def test_train_and_eval_baseline_svm(mini_dataset):
    df_train, df_val, df_test = mini_dataset
    pipe, res = train_and_eval_model("svm", df_train, df_val, df_test, seed=42)

    assert "val" in res
    assert "test" in res
    assert 0.0 <= res["test"]["accuracy"] <= 1.0


def test_save_and_predict_baseline(tmp_path, mini_dataset):
    df_train, df_val, df_test = mini_dataset
    pipe, res = train_and_eval_model("logreg", df_train, df_val, df_test, seed=42)

    out_models = tmp_path / "models"
    out_reports = tmp_path / "reports"
    save_baseline_artifacts(pipe, res, "logreg", out_dir=out_models, reports_dir=out_reports)

    model_file = out_models / "baseline_logreg.joblib"
    metrics_file = out_reports / "baseline_metrics.csv"
    assert model_file.exists()
    assert metrics_file.exists()

    # Probar predict_texts
    preds = predict_texts(model_file, ["hoy es un lindo día", "me siento muy triste y mal"])
    assert len(preds) == 2
    assert "label_pred" in preds[0]
    assert "probabilities" in preds[0]
    assert "confidence" in preds[0]

    # Probar predict_parquet
    in_parquet = tmp_path / "input.parquet"
    out_parquet = tmp_path / "output.parquet"
    pd.DataFrame({"text_clean": ["estoy contento", "depresión total"]}).to_parquet(in_parquet)

    predict_parquet(model_file, in_parquet, out_parquet)
    assert out_parquet.exists()
    df_out = pd.read_parquet(out_parquet)
    assert "label_pred" in df_out.columns
    assert len(df_out) == 2


def test_beto_label_mapping():
    labels = [0, 2, 0, 2, 0]
    label2id, id2label = get_label_mapping(labels)
    assert label2id == {0: 0, 2: 1}
    assert id2label == {0: 0, 1: 2}


class DummyTokenizer:
    """Mock rápido de tokenizer para testear TextClassificationDataset sin descargar 400MB."""

    def __call__(self, text, truncation=True, padding=True, max_length=16, return_tensors="pt"):
        return {
            "input_ids": torch.tensor([[101, 102] + [0] * (max_length - 2)]),
            "attention_mask": torch.tensor([[1, 1] + [0] * (max_length - 2)]),
        }


def test_beto_text_dataset():
    texts = ["hola mundo", "segundo texto"]
    labels = [0, 1]
    tokenizer = DummyTokenizer()
    ds = TextClassificationDataset(texts, labels, tokenizer, max_length=16)

    assert len(ds) == 2
    item = ds[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["labels"].item() == 0
    assert item["input_ids"].shape == (16,)
