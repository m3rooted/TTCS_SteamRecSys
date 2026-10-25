"""Content-based recommendations from Steam game metadata."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack


ALIASES = {
    "appid": ("appid", "app id", "app_id"),
    "name": ("name",),
    "release_date": ("release date", "release_date"),
    "estimated_owners": ("estimated owners", "estimated_owners"),
    "peak_ccu": ("peak ccu", "peak_ccu"),
    "required_age": ("required age", "required_age"),
    "price": ("price",),
    "about_the_game": ("about the game", "about_the_game", "detailed description", "detailed_description"),
    "supported_languages": ("supported language", "supported languages", "supported_languages"),
    "reviews": ("reviews",),
    "windows": ("windows",),
    "positive": ("positive",),
    "recommendations": ("recommendations",),
    "average_playtime_forever": ("average playtime forever", "average_playtime_forever"),
    "developers": ("developers",),
    "publishers": ("publishers",),
    "categories": ("categories",),
    "genres": ("genres",),
    "tags": ("tags",),
}

TEXT_FIELDS = (
    "name", "about_the_game", "reviews", "supported_languages", "developers",
    "publishers", "categories", "genres", "tags",
)
NUMERIC_FIELDS = (
    "release_date", "estimated_owners", "peak_ccu", "required_age", "price",
    "positive", "recommendations", "average_playtime_forever", "windows",
)
TOKEN_RE = re.compile(r"[\w+#]{2,}", re.UNICODE)


@dataclass(frozen=True)
class GameCatalog:
    frame: pd.DataFrame

    @property
    def n_games(self) -> int:
        return len(self.frame)


def _normalize_header(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def load_game_catalog(path: str) -> GameCatalog:
    frame = pd.read_csv(path, low_memory=False)
    by_normalized = {_normalize_header(column): column for column in frame.columns}
    rename: dict[str, str] = {}
    missing: list[str] = []
    for canonical, aliases in ALIASES.items():
        source = next((by_normalized[_normalize_header(alias)] for alias in aliases
                       if _normalize_header(alias) in by_normalized), None)
        if source is None:
            missing.append(canonical)
        else:
            rename[source] = canonical
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    frame = frame.rename(columns=rename)[list(ALIASES)].copy()
    frame["name"] = frame["name"].fillna("").astype(str).str.strip()
    frame = frame[frame["name"].ne("")].drop_duplicates("appid", keep="first")
    if frame.empty:
        raise ValueError("No games with a name were found in the dataset")
    frame = frame.reset_index(drop=True)
    return GameCatalog(frame)


def _tokens(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = unicodedata.normalize("NFKC", str(value)).lower()
    return TOKEN_RE.findall(text)


def _field_tfidf(values: pd.Series, prefix: str = "") -> csr_matrix:
    documents = [_tokens(value) for value in values]
    vocabulary: dict[str, int] = {}
    document_frequency: dict[str, int] = {}
    for tokens in documents:
        seen = set(tokens)
        for token in seen:
            document_frequency[token] = document_frequency.get(token, 0) + 1
    for token in sorted(document_frequency):
        vocabulary[token] = len(vocabulary)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    n_docs = max(len(documents), 1)
    for row, tokens in enumerate(documents):
        if not tokens:
            continue
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            column = vocabulary[token]
            tf = 1.0 + np.log(count)
            idf = np.log((1.0 + n_docs) / (1.0 + document_frequency[token])) + 1.0
            rows.append(row)
            cols.append(column)
            vals.append(tf * idf)
    matrix = csr_matrix((vals, (rows, cols)), shape=(n_docs, len(vocabulary)), dtype=np.float32)
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    matrix = matrix.multiply((1.0 / norms)[:, None]).tocsr()
    if prefix:
        # Field matrices are concatenated, so each field has its own vocabulary already.
        return matrix
    return matrix


def _owners_midpoint(value: object) -> float:
    if pd.isna(value):
        return 0.0
    numbers = re.findall(r"\d[\d,]*", str(value))
    if not numbers:
        return 0.0
    parsed = [float(number.replace(",", "")) for number in numbers]
    return float(np.mean(parsed))


def _numeric_matrix(frame: pd.DataFrame) -> csr_matrix:
    columns: list[np.ndarray] = []
    for field in NUMERIC_FIELDS:
        values = frame[field]
        if field == "estimated_owners":
            numeric = values.map(_owners_midpoint).to_numpy(dtype=np.float64)
        elif field == "release_date":
            numeric = pd.to_datetime(values, errors="coerce").dt.year.fillna(0).to_numpy(dtype=np.float64)
        elif field == "windows":
            numeric = values.astype(str).str.strip().str.lower().map(
                {"true": 1.0, "1": 1.0, "yes": 1.0, "false": 0.0, "0": 0.0, "no": 0.0}
            ).fillna(0.0).to_numpy(dtype=np.float64)
        else:
            numeric = pd.to_numeric(values.astype(str).str.replace(",", "", regex=False), errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        numeric = np.log1p(np.maximum(numeric, 0.0))
        maximum = float(numeric.max(initial=0.0))
        if maximum:
            numeric = numeric / maximum
        columns.append(numeric.astype(np.float32))
    return csr_matrix(np.column_stack(columns), dtype=np.float32)


@dataclass
class ContentBasedRecommender:
    catalog: GameCatalog
    features: csr_matrix

    @classmethod
    def fit(cls, catalog: GameCatalog) -> "ContentBasedRecommender":
        blocks = [_field_tfidf(catalog.frame[field]) / np.sqrt(len(TEXT_FIELDS)) for field in TEXT_FIELDS]
        numeric = _numeric_matrix(catalog.frame) / np.sqrt(len(NUMERIC_FIELDS))
        # Give each metadata field a comparable contribution regardless of vocabulary size.
        features = hstack([*blocks, numeric], format="csr", dtype=np.float32)
        row_norms = np.sqrt(np.asarray(features.multiply(features).sum(axis=1)).ravel())
        row_norms[row_norms == 0] = 1.0
        features = features.multiply((1.0 / row_norms)[:, None]).tocsr()
        return cls(catalog, features)

    def find_game(self, query: str) -> int:
        frame = self.catalog.frame
        query = str(query).strip()
        appid_match = frame.index[frame["appid"].astype(str).eq(query)].tolist()
        if appid_match:
            return int(appid_match[0])
        exact = frame.index[frame["name"].str.casefold().eq(query.casefold())].tolist()
        if exact:
            return int(exact[0])
        partial = frame.index[frame["name"].str.casefold().str.contains(re.escape(query.casefold()), regex=True)].tolist()
        if len(partial) == 1:
            return int(partial[0])
        if partial:
            examples = ", ".join(frame.iloc[partial[:8]]["name"].tolist())
            raise ValueError(f"Game name is ambiguous. Matches: {examples}")
        raise ValueError(f"No game matched {query!r}")

    def recommend(self, game: str, top_k: int = 10) -> pd.DataFrame:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        seed_idx = self.find_game(game)
        scores = (self.features[seed_idx] @ self.features.T).toarray().ravel()
        scores[seed_idx] = -1.0
        count = min(top_k, self.catalog.n_games - 1)
        best = np.argsort(-scores, kind="stable")[:count]
        result = self.catalog.frame.iloc[best].copy()
        result.insert(0, "similarity", scores[best])
        return result[["similarity", "appid", "name", "genres", "categories", "tags", "price"]].reset_index(drop=True)
