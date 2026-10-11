"""Project configuration: project.toml, with defaults, resolved to absolute paths."""

from __future__ import annotations

import copy
import os
import sys
import tomllib

DEFAULTS: dict = {
    "project": {"name": "", "contact_email": ""},
    "corpus": {
        "seeds": "seeds.csv",
        "min_year": 2015,
        "max_backward_per_seed": 600,
        "max_forward_per_seed": 600,
        "min_title_similarity": 0.72,
        "keep_types": ["article", "preprint", "book-chapter", "book", "review",
                       "proceedings-article"],
        "language": "en",
        # Roles whose seeds define the review's core. Candidates linked to one are
        # flagged `linked_to_core` and ranked first.
        "core_roles": [],
        # Seeds (by `ref`) whose OpenAlex topics define the target domain T used
        # to weight seeds by topical precision. Empty = the seeds of core_roles.
        "target_refs": [],
        "topic_score_min": 0.3,
        "weight_by_precision": True,
        "precision_min_sample": 30,
        "forward_sample_size": 200,
        "forward_sample_seed": 42,
        # Roles whose own reference lists are checked for coverage by the corpus.
        "coverage_roles": [],
        "backfill_abstracts": True,
    },
    "eligibility": {
        "require_open_access": True,
        "exclude_doi_prefixes": [],
    },
    "model": {"id": "claude-opus-5-5"},
    "screening": {
        "criteria": "criteria.md",
        "reason_words": 40,
        "reason_fail_words": 50,
        "max_usd": 20.0,
        "calibration_per_stratum": 25,
        "calibration_seed": 20260928,
        "kappa_floor": 0.70,
        "recall_floor": 0.95,
    },
    "extraction": {
        "quote_words": 25,
        "max_usd": 20.0,
        "assumed_output_tokens": 9000,
        "schemas": {},
    },
    "report": {"template": "report_template.md", "crosstabs": []},
}


class Config:
    def __init__(self, root: str, data: dict):
        self.root = os.path.abspath(root)
        self.data = data

    def __getitem__(self, key: str) -> dict:
        return self.data[key]

    def path(self, *parts: str) -> str:
        """A path inside the project directory."""
        return os.path.join(self.root, *parts)

    # Fixed output layout, so every stage finds the previous one's output.
    @property
    def data_dir(self) -> str:
        return self.path("data")

    @property
    def provenance(self) -> str:
        return self.path("data", "provenance.json")

    @property
    def candidates(self) -> str:
        return self.path("data", "candidates.csv")

    @property
    def eligibility(self) -> str:
        return self.path("data", "eligibility.csv")

    @property
    def mailto(self) -> str:
        m = self["project"].get("contact_email", "")
        if not m:
            sys.exit("project.toml: set [project] contact_email (OpenAlex and "
                     "Unpaywall ask for one; it is sent as a polite-pool contact).")
        return m


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(root: str = ".") -> Config:
    path = os.path.join(root, "project.toml")
    if not os.path.exists(path):
        sys.exit(f"no project.toml in {os.path.abspath(root)} -- run `litreview init <dir>` "
                 f"or pass --project <dir>")
    with open(path, "rb") as f:
        user = tomllib.load(f)
    return Config(root, _merge(DEFAULTS, user))
