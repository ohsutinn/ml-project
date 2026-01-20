import json
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Preprocessor:
    """
    전처리 상태를 기반으로 inference 입력을 학습 시점의 feature 스키마로 맞춘다.
    - drop_columns: 학습 시 제외했던 컬럼
    - categorical_columns: 원-핫 인코딩 대상 컬럼
    - feature_columns: 학습 모델 입력 순서 (label 제외)
    """

    feature_columns: List[str]
    label_column: Optional[str]
    drop_columns: List[str]
    categorical_columns: List[str]

    @classmethod
    def from_state_file(cls, path: str) -> "Preprocessor":
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        return cls.from_state_dict(state)

    @classmethod
    def from_state_dict(cls, state: dict) -> "Preprocessor":
        schema_version = state.get("schema_version")
        if schema_version is None:
            raise ValueError("preprocess_state에 schema_version이 필요합니다.")

        required_fields = [
            "feature_columns",
            "categorical_columns",
            "drop_columns",
        ]
        for field in required_fields:
            if field not in state:
                raise ValueError(f"preprocess_state에 {field} 필드가 필요합니다.")

        feature_columns = list(state.get("feature_columns", []))
        categorical_columns = list(state.get("categorical_columns", []))
        drop_columns = list(state.get("drop_columns", []))
        label_column = state.get("label_column")

        if label_column and label_column in feature_columns:
            raise ValueError("preprocess_state의 feature_columns에 label_column이 포함되어 있습니다.")
        if len(feature_columns) != len(set(feature_columns)):
            raise ValueError("preprocess_state의 feature_columns에 중복 컬럼이 있습니다.")
        if len(categorical_columns) != len(set(categorical_columns)):
            raise ValueError("preprocess_state의 categorical_columns에 중복 컬럼이 있습니다.")

        return cls(
            feature_columns=feature_columns,
            label_column=label_column,
            drop_columns=drop_columns,
            categorical_columns=categorical_columns,
        )

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        df = data.copy()

        # 학습 때 제거한 컬럼 drop
        if self.drop_columns:
            df = df.drop(columns=self.drop_columns, errors="ignore")

        # label 컬럼이 들어오면 제거 (서빙 입력은 feature만 사용)
        if self.label_column and self.label_column in df.columns:
            df = df.drop(columns=[self.label_column])

        cat_cols = [c for c in self.categorical_columns if c in df.columns]
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

        # 학습 시 feature 스키마에 맞춰 재정렬/결측 0 채움
        df = df.reindex(columns=self.feature_columns, fill_value=0)
        return df.to_numpy(dtype="float32")


# Bento PicklableModel 이 import 시 경로로 지정된 root 에 클래스가 있어야 해서 파일에 남겨둔다.
__all__ = ["Preprocessor"]
