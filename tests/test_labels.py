import subprocess

from skills_builder.config import TAU2_COMMIT, TAU2_REPO_DIR
from skills_builder.labels import label_counts, load_labels

EXPECTED_TRAIN = {
    "modify_items": 20, "return": 23, "exchange": 19, "cancel": 13,
    "modify_order_address": 13, "modify_user_address": 6, "transfer": 2,
    "modify_payment": 0, "answer": 25,
}


def test_tau2_submodule_is_pinned():
    head = subprocess.run(
        ["git", "-C", str(TAU2_REPO_DIR), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head == TAU2_COMMIT


def test_split_sizes():
    labels = load_labels()
    assert len(labels) == 114
    assert sum(r["split"] == "train" for r in labels.values()) == 74
    assert sum(r["split"] == "test" for r in labels.values()) == 40


def test_train_label_counts():
    assert label_counts(load_labels(), "train") == EXPECTED_TRAIN


def test_committed_labels_match_tau2():
    from skills_builder.labels import build_labels

    assert build_labels() == load_labels()
