"""
Name Normalization System

Handles common name variations, especially those from speech-to-text systems.
Ensures consistent entity naming across the knowledge graph.
"""

from typing import Dict, Optional


# Canonical name mappings (STT variation -> Correct name)
NAME_MAPPINGS = {
    # Alia's name (STT often transcribes as "Aaliyah")
    'aaliyah': 'Alia',
    'aliyah': 'Alia',
    'aaliya': 'Alia',
    'aliya': 'Alia',

    # Add other common variations here as needed
    # 'mike': 'Michael',
    # 'jon': 'John',
}


def normalize_name(name: str) -> str:
    """
    Normalize a name to its canonical form

    Args:
        name: Name as it appears (possibly from STT)

    Returns:
        Canonical name

    Example:
        normalize_name("Aaliyah") -> "Alia"
        normalize_name("Alia") -> "Alia"
        normalize_name("John") -> "John"
    """
    if not name:
        return name

    # Check if this is a known variation
    name_lower = name.lower().strip()

    if name_lower in NAME_MAPPINGS:
        return NAME_MAPPINGS[name_lower]

    # Not a known variation - return as-is (but cleaned)
    return name.strip()


def get_name_variants(canonical_name: str) -> list[str]:
    """
    Get all known variants of a canonical name

    Args:
        canonical_name: The correct canonical form

    Returns:
        List of all known variants (including canonical)

    Example:
        get_name_variants("Alia") -> ["Alia", "Aaliyah", "Aliyah", "Aaliya", "Aliya"]
    """
    variants = [canonical_name]

    # Find all mappings that point to this canonical name
    canonical_lower = canonical_name.lower()

    for variant, canonical in NAME_MAPPINGS.items():
        if canonical.lower() == canonical_lower:
            # Add the properly cased variant
            variants.append(canonical)
            # Also add the raw variant key (in case it's different casing)
            if variant not in [v.lower() for v in variants]:
                variants.append(variant.title())

    return list(set(variants))


def should_normalize_name(name: str) -> bool:
    """
    Check if a name has known variations that need normalization

    Args:
        name: Name to check

    Returns:
        True if this name should be normalized
    """
    return name.lower().strip() in NAME_MAPPINGS


# Precomputed reverse mapping for efficient lookups
CANONICAL_TO_VARIANTS: Dict[str, list[str]] = {}

def _build_reverse_mapping():
    """Build reverse mapping for efficient variant lookups"""
    global CANONICAL_TO_VARIANTS

    # Clear existing
    CANONICAL_TO_VARIANTS.clear()

    # Build reverse mapping
    for variant, canonical in NAME_MAPPINGS.items():
        if canonical not in CANONICAL_TO_VARIANTS:
            CANONICAL_TO_VARIANTS[canonical] = [canonical]

        # Add variant with proper casing
        variant_proper = variant.title()
        if variant_proper not in CANONICAL_TO_VARIANTS[canonical]:
            CANONICAL_TO_VARIANTS[canonical].append(variant_proper)

# Build on module load
_build_reverse_mapping()


if __name__ == "__main__":
    # Test the normalization
    print("Testing name normalization:")
    print("=" * 60)

    test_names = [
        "Aaliyah",
        "Alia",
        "AALIYAH",
        "aaliyah",
        "Aliyah",
        "James",  # Should pass through unchanged
        "John",   # Should pass through unchanged
    ]

    for name in test_names:
        normalized = normalize_name(name)
        is_variant = should_normalize_name(name)
        print(f"{name:15} -> {normalized:15} (variant: {is_variant})")

    print("\n" + "=" * 60)
    print("Variants for 'Alia':")
    variants = get_name_variants("Alia")
    for v in variants:
        print(f"  - {v}")
