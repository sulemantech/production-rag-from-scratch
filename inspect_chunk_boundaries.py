"""
Search the FULL corpus (not just the retrieved pool) for chunks containing
key terms tied to specific failing-question facts, then print each match
plus its immediate neighbors (same subject/grade, adjacent list index --
chunking is sequential within a book, so neighbors are physically adjacent
text). Goal: see whether the needed fact is split across a chunk boundary,
buried inside unrelated exercise text, or genuinely absent.
"""
import json

with open("datasets/mdcat_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)


def show_matches(term_groups, subject, label, context=1):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    found_any = False
    for i, c in enumerate(chunks):
        if c["subject"] != subject:
            continue
        text_lower = c["text"].lower()
        if all(any(t in text_lower for t in group) for group in term_groups):
            found_any = True
            print(f"\n--- MATCH at index {i} (grade={c.get('grade')}) ---")
            print(c["text"][:400].replace("\n", " "))
            for off in range(-context, context + 1):
                if off == 0:
                    continue
                j = i + off
                if 0 <= j < len(chunks) and chunks[j]["subject"] == subject:
                    print(f"\n  [neighbor idx {j}, offset {off:+d}]")
                    print("  " + chunks[j]["text"][:250].replace("\n", " "))
    if not found_any:
        print("(no chunk matched all term groups at all)")


# Q10: Coulomb's constant k depends on medium + system of units
show_matches(
    [["coulomb"], ["medium", "dielectric"], ["unit"]],
    "Physics",
    "Q10: Coulomb's constant k depends upon nature of medium and system of units",
)

# Q23: pair with all-unpaired vs all-paired d orbitals -> Cr and Zn
show_matches(
    [["zn", "zinc"], ["3d10", "d10", "3d 10"]],
    "Chemistry",
    "Q23: Zn electron configuration (all paired d10)",
)
show_matches(
    [["cr", "chromium"], ["3d5", "d5", "unpaired"]],
    "Chemistry",
    "Q23: Cr electron configuration (all unpaired d5)",
)

# Q14: motional EMF of a rod moving in a magnetic field
show_matches(
    [["rod"], ["magnetic field"], ["emf", "e.m.f"]],
    "Physics",
    "Q14: motional EMF of a rod moving through a magnetic field",
)
