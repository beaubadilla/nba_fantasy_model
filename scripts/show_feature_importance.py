import joblib
import pandas as pd
from pathlib import Path

model_path = Path("outputs/models/xgb_pipeline.joblib")
if not model_path.exists():
    raise SystemExit(
        "Model artifact not found at outputs/models/xgb_pipeline.joblib. Run training first."
    )

pipeline = joblib.load(model_path)
model = pipeline.named_steps.get("model")
preprocess = pipeline.named_steps.get("preprocess")
if model is None or preprocess is None:
    raise SystemExit("Pipeline missing preprocess or model steps.")

try:
    feature_names = preprocess.get_feature_names_out()
except Exception as exc:
    feature_names = [f"f{i}" for i in range(len(model.feature_importances_))]
    print(f"Warning: failed to recover feature names ({exc!r}). Using generic names.")

importances = model.feature_importances_
if importances.shape[0] != len(feature_names):
    raise SystemExit("Feature name count mismatch with importances.")

fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False)

with pd.option_context("display.max_rows", 100, "display.max_colwidth", None):
    print(fi_df.head(25).to_string(index=False))
