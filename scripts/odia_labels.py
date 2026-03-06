# scripts/odia_labels.py

# ─────────────────────────────────────────────────────────────
# Odia Translation Dictionary for Sign Language Classes
# All 76 classes translated and organized by category
# ─────────────────────────────────────────────────────────────


# ── 📅 Time of Day ────────────────────────────────────────────
TIME_OF_DAY = {
    "MORNING"   : "ସକାଳ",
    "AFTERNOON" : "ଅପରାହ୍ନ",
    "EVENING"   : "ସନ୍ଧ୍ୟା",
    "NIGHT"     : "ରାତ୍ରି",
    "TODAY"     : "ଆଜି",
    "TOMORROW"  : "କାଲି",
    "YESTERDAY" : "ଗତକାଲି",
    "TIME"      : "ସମୟ",
}

# ── 📆 Days of the Week ───────────────────────────────────────
DAYS_OF_WEEK = {
    "MONDAY"    : "ସୋମବାର",
    "TUESDAY"   : "ମଙ୍ଗଳବାର",
    "WEDNESDAY" : "ବୁଧବାର",
    "THURSDAY"  : "ଗୁରୁବାର",
    "FRIDAY"    : "ଶୁକ୍ରବାର",
    "SATURDAY"  : "ଶନିବାର",
    "SUNDAY"    : "ରବିବାର",
}

# ── ⏱️ Time Units ─────────────────────────────────────────────
TIME_UNITS = {
    "SECOND"    : "ସେକେଣ୍ଡ",
    "MINUTE"    : "ମିନିଟ",
    "HOUR"      : "ଘଣ୍ଟା",
    "WEEK"      : "ସପ୍ତାହ",
    "MONTH"     : "ମାସ",
    "YEAR"      : "ବର୍ଷ",
}

# ── 🐾 Animals ────────────────────────────────────────────────
ANIMALS = {
    "ANIMAL"    : "ପ୍ରାଣୀ",
    "BIRD"      : "ପକ୍ଷୀ",
    "CAT"       : "ବିଲେଇ",
    "COW"       : "ଗାଈ",
    "DOG"       : "କୁକୁର",
    "FISH"      : "ମାଛ",
    "HORSE"     : "ଘୋଡା",
    "MOUSE"     : "ମୂଷା",
}

# ── 👗 Clothing & Accessories ─────────────────────────────────
CLOTHING = {
    "CLOTHING"  : "ପୋଷାକ",
    "DRESS"     : "ଡ୍ରେସ",
    "HAT"       : "ଟୋପି",
    "PANT"      : "ପ୍ୟାଣ୍ଟ",
    "POCKET"    : "ପକେଟ",
    "SHIRT"     : "ସାର୍ଟ",
    "SHOES"     : "ଜୋତା",
    "SKIRT"     : "ସ୍କର୍ଟ",
    "SUIT"      : "ସୁଟ",
    "T_SHIRT"   : "ଟି-ସାର୍ଟ",
}

# ── 😊 Emotions & Character ───────────────────────────────────
EMOTIONS = {
    "BAD"       : "ଖରାପ",
    "BEAUTIFUL" : "ସୁନ୍ଦର",
    "FAMOUS"    : "ପ୍ରସିଦ୍ଧ",
    "GOOD"      : "ଭଲ",
    "HAPPY"     : "ଖୁସି",
    "SAD"       : "ଦୁଃଖୀ",
    "UGLY"      : "କୁରୂପ",
}

# ── 📐 Physical Descriptions ──────────────────────────────────
PHYSICAL = {
    "BIG"       : "ବଡ",
    "CURVED"    : "ବଙ୍କା",
    "FLAT"      : "ସମତଳ",
    "LONG"      : "ଲମ୍ବା",
    "LOOSE"     : "ଢିଲା",
    "NARROW"    : "ସଙ୍କୀର୍ଣ୍ଣ",
    "SHORT"     : "ଛୋଟ",
    "SMALL"     : "ଛୋଟ",
    "TALL"      : "ଉଚ୍ଚ",
    "WIDE"      : "ଚଉଡା",
}

# ── 🌡️ Sensory & Weather ──────────────────────────────────────
SENSORY = {
    "COLD"      : "ଥଣ୍ଡା",
    "DRY"       : "ଶୁଖିଲା",
    "HOT"       : "ଗରମ",
    "LIGHT"     : "ହାଲୁକା",
    "LOUD"      : "ଜୋରରେ",
    "QUIET"     : "ଶାନ୍ତ",
    "WARM"      : "ଉଷ୍ଣ",
    "WET"       : "ଓଦା",
}

# ── 🏥 Health & People ────────────────────────────────────────
HEALTH_PEOPLE = {
    "BLIND"     : "ଅନ୍ଧ",
    "DEAF"      : "ବଧିର",
    "FEMALE"    : "ମହିଳା",
    "HEALTHY"   : "ସୁସ୍ଥ",
    "OLD"       : "ବୃଦ୍ଧ",
    "SICK"      : "ଅସୁସ୍ଥ",
    "YOUNG"     : "ଯୁବ",
}

# ── ⚡ Speed & Value ──────────────────────────────────────────
SPEED_VALUE = {
    "CHEAP"     : "ଶସ୍ତା",
    "EXPENSIVE" : "ମହଙ୍ଗା",
    "FAST"      : "ଦ୍ରୁତ",
    "SLOW"      : "ଧୀର",
}


# ─────────────────────────────────────────────────────────────
# Master dictionary — merge all categories
# ─────────────────────────────────────────────────────────────
ODIA_LABELS = {
    **TIME_OF_DAY,
    **DAYS_OF_WEEK,
    **TIME_UNITS,
    **ANIMALS,
    **CLOTHING,
    **EMOTIONS,
    **PHYSICAL,
    **SENSORY,
    **HEALTH_PEOPLE,
    **SPEED_VALUE,
}


# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────
def get_odia(english_label: str) -> str:
    """
    Returns Odia translation for a given English sign label.
    Falls back to the English label if not found.
    """
    return ODIA_LABELS.get(english_label.upper(), english_label)


def get_category(english_label: str) -> str:
    """Returns the category name for a given label."""
    categories = {
        "📅 Time of Day"     : TIME_OF_DAY,
        "📆 Days of Week"    : DAYS_OF_WEEK,
        "⏱️ Time Units"      : TIME_UNITS,
        "🐾 Animals"         : ANIMALS,
        "👗 Clothing"        : CLOTHING,
        "😊 Emotions"        : EMOTIONS,
        "📐 Physical"        : PHYSICAL,
        "🌡️ Sensory/Weather" : SENSORY,
        "🏥 Health & People" : HEALTH_PEOPLE,
        "⚡ Speed & Value"   : SPEED_VALUE,
    }
    for category, labels in categories.items():
        if english_label.upper() in labels:
            return category
    return "Unknown"


# ─────────────────────────────────────────────────────────────
# Quick verify — run: python scripts/odia_labels.py
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'─'*45}")
    print(f"  Total classes mapped: {len(ODIA_LABELS)}")
    print(f"{'─'*45}\n")

    categories = {
        "📅 Time of Day"     : TIME_OF_DAY,
        "📆 Days of Week"    : DAYS_OF_WEEK,
        "⏱️ Time Units"      : TIME_UNITS,
        "🐾 Animals"         : ANIMALS,
        "👗 Clothing"        : CLOTHING,
        "😊 Emotions"        : EMOTIONS,
        "📐 Physical"        : PHYSICAL,
        "🌡️ Sensory/Weather" : SENSORY,
        "🏥 Health & People" : HEALTH_PEOPLE,
        "⚡ Speed & Value"   : SPEED_VALUE,
    }

    for cat_name, cat_dict in categories.items():
        print(f"{cat_name}  ({len(cat_dict)} words)")
        for eng, odia in cat_dict.items():
            print(f"   {eng:<15} →  {odia}")
        print()