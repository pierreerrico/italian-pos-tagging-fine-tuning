from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import json

ROOT = Path(__file__).resolve().parents[2]
KEYS = ("sentence_index", "sent_id", "word_index", "word", "true_label")
TAGS = set("ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X".split())


def read_predictions(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set((*KEYS, "predicted_label", "correct")).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing prediction columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty predictions: {path}")
    previous_sentence, previous_word, sentence_id = -1, -1, None
    for row in rows:
        sid, wid = int(row["sentence_index"]), int(row["word_index"])
        if sid == previous_sentence:
            valid_order = wid == previous_word + 1 and row["sent_id"] == sentence_id
        else:
            valid_order = sid == previous_sentence + 1 and wid == 0
            sentence_id = row["sent_id"]
        if not valid_order or not row["word"] or not sentence_id:
            raise ValueError(f"Invalid sentence/word order: {path}, position {sid}:{wid}")
        if row["true_label"] not in TAGS or row["predicted_label"] not in TAGS | {"<INVALID>"}:
            raise ValueError(f"Unknown tag: {path}, position {sid}:{wid}")
        expected = str(row["true_label"] == row["predicted_label"]).lower()
        if row["correct"].lower() != expected:
            raise ValueError(f"Inconsistent correctness flag: {path}, position {sid}:{wid}")
        previous_sentence, previous_word = sid, wid
    return rows


def extract(runs, limit):
    paths = sorted(path for model in ("bert", "electra", "geppetto")
                   for path in (runs / model).glob("*/seed_*/predictions.csv"))
    if not paths:
        raise ValueError(f"No saved predictions in {runs}; run the notebook experiments first")
    reference, confusions, examples, sources = None, [], [], []
    for path in paths:
        model, variant, seed_directory, _ = path.relative_to(runs).parts
        seed = int(seed_directory.removeprefix("seed_"))
        if model not in {"bert", "electra", "geppetto"} or variant not in {"original", "lowercase"}:
            raise ValueError(f"Unknown model or variant: {path}")
        rows = read_predictions(path)
        signature = [tuple(row[key] for key in KEYS) for row in rows]
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f"Gold words or annotations differ between runs: {path}")
        sentences = defaultdict(list)
        for row in rows:
            sentences[row["sentence_index"]].append(row["word"])
        wrong = [row for row in rows if row["true_label"] != row["predicted_label"]]
        counts = Counter((row["true_label"], row["predicted_label"]) for row in wrong)
        identity = {"model": model, "variant": variant, "seed": seed}
        for (actual, predicted), count in counts.most_common():
            confusions.append({**identity, "true_label": actual, "predicted_label": predicted, "errors": count})
        for row in wrong[:limit]:
            words = sentences[row["sentence_index"]]
            examples.append({
                **identity, "sentence_index": int(row["sentence_index"]), "sent_id": row["sent_id"],
                "word_index": int(row["word_index"]), "word": row["word"],
                "true_label": row["true_label"], "predicted_label": row["predicted_label"],
                "sentence": " ".join(words),
                "input_sentence": " ".join(words if variant == "original" else [word.lower() for word in words]),
            })
        sources.append({**identity, "path": path.relative_to(runs).as_posix(),
                        "words": len(rows), "errors": len(wrong)})
    return confusions, examples, sources


def main():
    parser = argparse.ArgumentParser(description="Extract UPOS confusions and contextual errors from saved predictions.")
    parser.add_argument("--runs", type=Path, default=ROOT / "outputs/upos_trainer_v1")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=5, help="First erroneous words per run, in corpus order")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    try:
        confusions, examples, sources = extract(args.runs, args.limit)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    output = args.out or args.runs / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    fields = ["model", "variant", "seed", "true_label", "predicted_label", "errors"]
    with (output / "confusions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(confusions)
    data = {
        "selection": f"First {args.limit} erroneous words per run, in corpus order; not a representative sample.",
        "context": "Sentences are reconstructed by joining gold word forms with spaces.",
        "runs": sources,
        "examples": examples,
    }
    (output / "error_examples.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Analysed {len(sources)} saved runs; saved {len(examples)} examples and {len(confusions)} confusion counts in {output}")


if __name__ == "__main__":
    main()
