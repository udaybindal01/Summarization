def evaluate_graph_extraction(gold_standard, predictions):
    """
    Calculates Precision, Recall, and F1-score for extracted triplets.
    
    :param gold_standard: dict mapping scene_id to a list of true triplets (subj, pred, obj)
    :param predictions: dict mapping scene_id to a list of predicted triplets (subj, pred, obj)
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for scene_id, gold_triplets in gold_standard.items():
        # Get predictions for this scene, default to empty list if missing
        pred_triplets = predictions.get(scene_id, [])

        # Normalize to lowercase tuples to avoid case-sensitivity penalties
        gold_set = set((s.lower(), p.lower(), o.lower()) for s, p, o in gold_triplets)
        pred_set = set((s.lower(), p.lower(), o.lower()) for s, p, o in pred_triplets)

        # Calculate True Positives, False Positives, and False Negatives
        tp = len(gold_set.intersection(pred_set))
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Prevent division by zero
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("--- Intrinsic Evaluation Results ---")
    print(f"Total True Positives (Correct Edges): {total_tp}")
    print(f"Total False Positives (Invented Edges): {total_fp}")
    print(f"Total False Negatives (Missed Edges): {total_fn}")
    print("-" * 34)
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1 * 100:.2f}%)")

    return precision, recall, f1

# ==========================================
# Example Usage for your MENSA subset
# ==========================================
if __name__ == "__main__":
    # 1. Your manual annotations (The "Gold Standard")
    gold_data = {
        "The_Ides_of_March_Scene_000": [
            ("Stephen", "snaps", "fingers"),
            ("director", "speaks", "speaker"),
            ("Stephen", "looks", "stagehands")
        ]
    }

    # 2. What your Python Daemon actually extracted
    predicted_data = {
        "The_Ides_of_March_Scene_000": [
            ("stephen", "snaps", "fingers"), # Correct (True Positive)
            ("director", "speaks", "speaker"), # Correct (True Positive)
            ("stephen", "looks", "around"), # Incorrect Object (False Positive & False Negative)
            ("lights", "come", "podium") # Extra extraction (False Positive)
        ]
    }

    evaluate_graph_extraction(gold_data, predicted_data)