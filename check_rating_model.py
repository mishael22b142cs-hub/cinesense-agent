import joblib
import pandas as pd

model = joblib.load("models/rating_model.pkl")

tests = [
    (5, 90, "Horror"),
    (20, 100, "Drama"),
    (50, 120, "Thriller"),
    (150, 130, "Animation"),
    (200, 150, "Science Fiction"),
]

for year in [2026, 2015, 2000]:   # 🔴 പരിധിക്ക് പുറത്തുള്ള year (2026) vs dataset-ന്റെ ഉള്ളിലുള്ള years
    rows = [{"budget_millions": b, "runtime": r, "year": year, "main_genre": g} for b, r, g in tests]
    preds = model.predict(pd.DataFrame(rows))
    print(f"\n📅 Year {year}:")
    for (b, r, g), p in zip(tests, preds):
        print(f"   {g:16} ${b:>3}M  {r} min  →  {p:.2f}")

rf = model.named_steps["rf"]
names = model.named_steps["prep"].get_feature_names_out()
top = sorted(zip(rf.feature_importances_, names), reverse=True)[:6]   # 🔴 model ഏറ്റവും കൂടുതൽ ആശ്രയിക്കുന്ന 6 features
print("\n⭐ Most important features:")
for imp, name in top:
    print(f"   {name:35} {imp:.2f}")