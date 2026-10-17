import sys
from pipeline import build_dataset, slugify
from insights import generate_report

sys.stdout.reconfigure(encoding="utf-8")
brand = " ".join(sys.argv[1:])
df = build_dataset(brand)
stats, qdf, text, notes = generate_report(brand, df)
table = stats.to_string(index=False)
print(table, "\n")
print(text)
print("\nFormat checks:", notes or "passed (ids, counts, tags only; claim support is not verified)")
print("\nEvidence:")
for r in qdf.itertuples():
    print(f"  [{r.id}] ({r.aspect}, {r.sentiment}) {r.text}")
with open(f"reports/report_{slugify(brand)}_v2.md", "w", encoding="utf-8") as f:
    f.write(f"# {brand}\n\n" + "\n".join("    " + l for l in table.splitlines())
            + f"\n\n{text}\n\nFormat checks: {notes or 'passed (claim support not verified)'}\n")
print(f"\nSaved reports/report_{slugify(brand)}_v2.md")
