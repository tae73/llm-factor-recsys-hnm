"""Super-Category-specific prompt templates and JSON schemas for L1+L2+L3 extraction.

Three Super-Categories derived from garment_group_name (21 unique values):
  - Apparel (~82%): Jersey Fancy/Basic, Knitwear, Trousers, Blouses, Dresses, etc.
  - Footwear (~7%): Shoes, Socks and Tights
  - Accessories (~11%): Accessories
  - Mixed: Unknown (3.7%) and Special Offers (1%) → routed by product_group_name fallback
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Super-Category Routing (garment_group_name → Super-Category)
# ---------------------------------------------------------------------------

# Primary mapping: garment_group_name (21 unique values in H&M data)
SUPER_CATEGORY_MAP: dict[str, str] = {
    # Apparel — garments and textiles
    "Jersey Fancy": "Apparel",
    "Jersey Basic": "Apparel",
    "Knitwear": "Apparel",
    "Under-, Nightwear": "Apparel",
    "Trousers": "Apparel",
    "Blouses": "Apparel",
    "Dresses Ladies": "Apparel",
    "Outdoor": "Apparel",
    "Trousers Denim": "Apparel",
    "Swimwear": "Apparel",
    "Shirts": "Apparel",
    "Woven/Jersey/Knitted mix Baby": "Apparel",
    "Shorts": "Apparel",
    "Dresses/Skirts girls": "Apparel",
    "Skirts": "Apparel",
    "Dressed": "Apparel",
    # Footwear
    "Shoes": "Footwear",
    "Socks and Tights": "Footwear",
    # Accessories
    "Accessories": "Accessories",
    # Mixed categories — routed by product_group_name fallback
    "Special Offers": "Mixed",
    "Unknown": "Mixed",
}

# Fallback mapping: product_group_name → Super-Category
# Used for "Mixed" garment groups (Unknown, Special Offers)
_PRODUCT_GROUP_SUPER_CATEGORY: dict[str, str] = {
    "Garment Upper body": "Apparel",
    "Garment Lower body": "Apparel",
    "Garment Full body": "Apparel",
    "Underwear": "Apparel",
    "Nightwear": "Apparel",
    "Underwear/nightwear": "Apparel",
    "Swimwear": "Apparel",
    "Shoes": "Footwear",
    "Socks & Tights": "Footwear",
    "Garment and Shoe care": "Footwear",
    "Accessories": "Accessories",
    "Bags": "Accessories",
    "Cosmetic": "Accessories",
    "Items": "Accessories",
    "Furniture": "Accessories",
    "Interior textile": "Accessories",
    "Stationery": "Accessories",
    "Unknown": "Apparel",  # Default for truly unknown
    "Fun": "Accessories",
}


def resolve_super_category(
    garment_group_name: str,
    product_group_name: str | None = None,
) -> str:
    """Resolve Super-Category with fallback for mixed garment groups.

    Returns "Apparel", "Footwear", or "Accessories".
    Falls back to "Apparel" if both lookups fail.
    """
    cat = SUPER_CATEGORY_MAP.get(garment_group_name)
    if cat is not None and cat != "Mixed":
        return cat
    # Fallback for Mixed (Unknown, Special Offers)
    if product_group_name:
        return _PRODUCT_GROUP_SUPER_CATEGORY.get(product_group_name, "Apparel")
    return "Apparel"

# ---------------------------------------------------------------------------
# Enum Values for Structured Output
# ---------------------------------------------------------------------------

# L2 enums (Universal across all categories)
STYLE_MOOD_VALUES = [
    "Casual",
    "Minimalist",
    "Classic",
    "Sporty",
    "Bohemian",
    "Romantic",
    "Edgy",
    "Preppy",
    "Streetwear",
    "Luxury",
    "Cozy",
    "Avant-garde",
    "Retro",
    "Feminine",
    "Masculine",
    "Androgynous",
    "Eclectic",
    "Natural",
    "Glam",
    "Utility",
    "Cottagecore",
    "Dark-academic",
    "Y2K",
    "Quiet-luxury",
]

OCCASION_VALUES = [
    "Everyday",
    "Work",
    "Formal",
    "Party",
    "Outdoor",
    "Lounge",
    "Active",
    "Date",
    "Travel",
    "Beach",
    "Wedding",
    "Festival",
    "School",
    "Ceremony",
]

TRENDINESS_VALUES = ["Classic", "Current", "Emerging", "Dated"]

SEASON_FIT_VALUES = ["Spring", "Summer", "Fall", "Winter", "All-season"]

# L3 enums (Shared)
COLOR_HARMONY_VALUES = [
    "Monochromatic",
    "Analogous",
    "Complementary",
    "Triadic",
    "Neutral",
    "Earth-tone",
    "Pastel",
    "Jewel-tone",
    "Neon",
]

TONE_SEASON_VALUES = [
    "Warm-Spring",
    "Warm-Autumn",
    "Cool-Summer",
    "Cool-Winter",
    "Neutral-Warm",
    "Neutral-Cool",
]

COORDINATION_ROLE_VALUES = [
    "Basic",
    "Statement",
    "Accent",
    "Layering",
    "Foundation",
    "Finishing",
]

# L3 Apparel-specific
SILHOUETTE_VALUES = [
    "A-line",
    "H-line",
    "I-line",
    "X-line",
    "V-line",
    "O-line",
    "Y-line",
    "Trapeze",
    "Cocoon",
    "Empire",
]

PROPORTION_EFFECT_VALUES = [
    "Lengthening",
    "Shortening",
    "Broadening",
    "Narrowing",
    "Balanced",
    "Volume-adding",
    "Streamlining",
]

# L3 Footwear-specific
FOOT_SILHOUETTE_VALUES = [
    "Streamlined",
    "Chunky",
    "Pointed",
    "Rounded",
    "Square",
    "Sculptural",
    "Flat-profile",
    "Platform",
]

HEIGHT_EFFECT_VALUES = [
    "Elevating",
    "Grounding",
    "Neutral",
    "Leg-lengthening",
    "Proportioning",
]

# L3 Accessories-specific
VISUAL_FORM_VALUES = [
    "Geometric",
    "Organic",
    "Structured",
    "Soft",
    "Minimalist",
    "Ornate",
    "Angular",
    "Fluid",
]

STYLING_EFFECT_VALUES = [
    "Focal-point",
    "Cohesion",
    "Contrast",
    "Balance",
    "Proportion",
    "Texture-layer",
    "Color-anchor",
]

# L1 Apparel-specific
NECKLINE_VALUES = [
    "Crew",
    "V-neck",
    "Scoop",
    "Boat",
    "Turtleneck",
    "Mock-neck",
    "Collar",
    "Henley",
    "Cowl",
    "Halter",
    "Off-shoulder",
    "Square",
    "Sweetheart",
    "Hooded",
    "Strapless",
    "N/A",
]

SLEEVE_TYPE_VALUES = [
    "Long",
    "Short",
    "3/4",
    "Sleeveless",
    "Cap",
    "Raglan",
    "Bell",
    "Puff",
    "Roll-up",
    "N/A",
]

FIT_VALUES = [
    "Slim",
    "Regular",
    "Relaxed",
    "Oversized",
    "Tailored",
    "Skinny",
    "Wide",
    "Loose",
    "Boxy",
    "Bodycon",
]

LENGTH_VALUES = [
    "Cropped",
    "Hip",
    "Waist",
    "Thigh",
    "Knee",
    "Midi",
    "Ankle",
    "Full-length",
    "Mini",
    "Maxi",
    "N/A",
]

# L1 Footwear-specific
TOE_SHAPE_VALUES = [
    "Round",
    "Pointed",
    "Square",
    "Almond",
    "Peep-toe",
    "Open-toe",
    "Cap-toe",
    "N/A",
]

SHAFT_HEIGHT_VALUES = [
    "Low-top",
    "Mid-top",
    "High-top",
    "Ankle",
    "Mid-calf",
    "Knee-high",
    "Over-knee",
    "N/A",
]

HEEL_TYPE_VALUES = [
    "Flat",
    "Low",
    "Mid",
    "High",
    "Stiletto",
    "Block",
    "Wedge",
    "Platform",
    "Kitten",
    "N/A",
]

SOLE_TYPE_VALUES = [
    "Rubber",
    "Leather",
    "Foam",
    "Cork",
    "Crepe",
    "Lug",
    "Commando",
    "Espadrille",
    "N/A",
]

# L1 Accessories-specific
FORM_FACTOR_VALUES = [
    "Small",
    "Medium",
    "Large",
    "Oversized",
    "Mini",
    "Compact",
    "Structured",
    "Soft",
    "Rigid",
]

SIZE_SCALE_VALUES = [
    "Petite",
    "Small",
    "Medium",
    "Large",
    "Oversized",
    "Adjustable",
    "One-size",
]

WEARING_METHOD_VALUES = [
    "Handheld",
    "Shoulder",
    "Crossbody",
    "Waist",
    "Wrist",
    "Neck",
    "Head",
    "Ear",
    "Finger",
    "Clip-on",
    "Pin",
    "Wrap",
    "Drape",
    "N/A",
]

PRIMARY_FUNCTION_VALUES = [
    "Storage",
    "Protection",
    "Decoration",
    "Warmth",
    "Support",
    "Fragrance",
    "Skincare",
    "Grooming",
    "Writing",
    "Home-decor",
    "Lighting",
]

# L1 Material enums (category-specific; detail goes in l1_material_detail)
MATERIAL_VALUES_APPAREL = [
    "Cotton", "Polyester", "Cotton blend", "Polyester blend",
    "Wool", "Wool blend", "Linen", "Linen blend", "Silk",
    "Nylon", "Viscose", "Denim", "Jersey", "Fleece",
    "Leather", "Faux leather", "Velvet", "Satin",
    "Chiffon", "Mesh", "Ribbed knit", "Other",
]
MATERIAL_VALUES_FOOTWEAR = [
    "Leather", "Faux leather", "Canvas", "Suede", "Faux suede",
    "Textile", "Rubber", "Synthetic", "Knit", "Mesh",
    "Patent leather", "Other",
]
MATERIAL_VALUES_ACCESSORIES = [
    "Metal", "Fabric", "Leather", "Faux leather", "Plastic",
    "Glass", "Wood", "Pearl", "Crystal", "Silicone",
    "Paper", "Ceramic", "Other",
]

# L1 Closure enums (category-specific)
CLOSURE_VALUES_APPAREL = [
    "Pullover", "Button", "Zipper", "Snap", "Hook-and-eye",
    "Drawstring", "Wrap", "Tie", "Velcro", "None", "N/A",
]
CLOSURE_VALUES_FOOTWEAR = [
    "Lace-up", "Slip-on", "Buckle", "Zip", "Velcro",
    "Elastic", "Strap", "Pull-on", "None",
]
CLOSURE_VALUES_ACCESSORIES = [
    "Clasp", "Snap", "Magnetic", "Tie", "Pin",
    "Buckle", "Hook", "Elastic", "Zip", "None", "N/A",
]

# L3 Style Lineage enum (normalized from 207 free-form → 45 values)
STYLE_LINEAGE_VALUES = [
    # Regional/Traditional
    "Scandinavian Minimalism", "French Chic", "Americana Prep",
    "British Tailoring", "Italian Luxury", "Japanese Avant-garde",
    "Korean Contemporary", "Mediterranean Coastal",
    # Subculture/Movement
    "Streetwear", "Athleisure", "Bohemian", "Grunge", "Punk",
    "Hip-Hop", "Skater", "Surf",
    # 2018-2020 Trends
    "Y2K", "Cottagecore", "Dark Academia", "Quiet Luxury",
    "Normcore", "Gorpcore", "Old Money",
    # Heritage/Workwear
    "Workwear Heritage", "Military Surplus", "Nautical", "Western",
    # Aesthetic/Theory
    "Art Deco", "Bauhaus Functional", "Mid-Century Modern",
    "Contemporary Minimalism", "Romantic Victorian", "Retro Sportswear",
    # Style Categories
    "Classic Formal", "Smart Casual", "Resort Wear",
    "Activewear Performance", "Loungewear Comfort",
    # Gender/Age-specific
    "Traditional Menswear", "Traditional Womenswear",
    "Kids Classic", "Teen Trend",
    # Material/Technique-centric
    "Denim Culture", "Knit Heritage", "Leather Craft",
]


# ---------------------------------------------------------------------------
# JSON Schemas (OpenAI Structured Output)
# ---------------------------------------------------------------------------


def _l2_schema_properties() -> dict:
    """L2 Perceptual attributes — universal across all categories."""
    return {
        "l2_style_mood": {
            "type": "array",
            "items": {"type": "string", "enum": STYLE_MOOD_VALUES},
            "minItems": 1,
            "maxItems": 3,
            "description": "Primary style/mood descriptors (1-3).",
        },
        "l2_occasion": {
            "type": "array",
            "items": {"type": "string", "enum": OCCASION_VALUES},
            "minItems": 1,
            "maxItems": 3,
            "description": "Suitable occasions (1-3).",
        },
        "l2_perceived_quality": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": (
                "Perceived quality WITHIN H&M range (1=budget basics e.g. Divided, "
                "2=everyday e.g. Basic collection, 3=standard, 4=premium line/CONSCIOUS, "
                "5=designer collab e.g. H&M x Balmain). Use the FULL 1-5 scale. "
                "Consider material weight, stitching quality, and brand positioning cues."
            ),
        },
        "l2_trendiness": {
            "type": "string",
            "enum": TRENDINESS_VALUES,
            "description": "Trend positioning.",
        },
        "l2_season_fit": {
            "type": "string",
            "enum": SEASON_FIT_VALUES,
            "description": "Best-fit season.",
        },
        "l2_target_impression": {
            "type": "string",
            "description": "One-phrase impression this item projects (e.g. 'effortless weekend').",
        },
        "l2_versatility": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "How many different outfits/contexts this item works with (1=niche, 5=universal).",
        },
    }


def _l3_shared_properties() -> dict:
    """L3 Theory attributes shared across all categories.

    Note: l3_tone_season is NOT included in the LLM schema — it is added
    post-extraction via rule-based COLOR_TO_TONE mapping in extractor.py.
    """
    return {
        "l3_color_harmony": {
            "type": "string",
            "enum": COLOR_HARMONY_VALUES,
            "description": "Color scheme classification.",
        },
        "l3_coordination_role": {
            "type": "string",
            "enum": COORDINATION_ROLE_VALUES,
            "description": "Role in outfit coordination.",
        },
        "l3_visual_weight": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": (
                "Visual weight based on FORM and VOLUME, not color or pattern. "
                "1=minimal structure (slim tee, thin scarf), 2=light (fitted shirt), "
                "3=moderate (regular blazer), 4=substantial (oversized coat, chunky knit), "
                "5=maximum volume (puffer jacket, ball gown). "
                "Rule: Slim/Skinny fit or I-line silhouette → max 3. "
                "Oversized/Loose/Boxy fit or O-line silhouette → min 3. "
                "Basic coordination role → max 3."
            ),
        },
        "l3_style_lineage": {
            "type": "array",
            "items": {"type": "string", "enum": STYLE_LINEAGE_VALUES},
            "minItems": 1,
            "maxItems": 2,
            "description": "Fashion movement heritage (select from allowed values).",
        },
    }


def _build_apparel_schema() -> dict:
    """Complete JSON schema for Apparel items (L1+L2+L3 = 21 fields)."""
    properties = {
        # L1 Product — Apparel
        "l1_material": {
            "type": "string",
            "enum": MATERIAL_VALUES_APPAREL,
            "description": "Primary material category. Put blend details in l1_material_detail.",
        },
        "l1_closure": {
            "type": "string",
            "enum": CLOSURE_VALUES_APPAREL,
            "description": "Closure type.",
        },
        "l1_design_details": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
            "description": (
                "Notable design details (at least 1 required). Include: "
                "surface finish, construction, decorative elements, "
                "graphic/print, hardware. E.g. 'ribbed texture', 'metal eyelets'."
            ),
        },
        "l1_material_detail": {
            "type": "string",
            "description": (
                "Material composition details: blend ratio, weave, finish. "
                "E.g. '60% cotton 40% polyester, brushed fleece finish', "
                "'100% organic cotton, enzyme-washed'. Write 'single material' if pure."
            ),
        },
        "l1_neckline": {
            "type": "string",
            "enum": NECKLINE_VALUES,
            "description": "Neckline style.",
        },
        "l1_sleeve_type": {
            "type": "string",
            "enum": SLEEVE_TYPE_VALUES,
            "description": "Sleeve style.",
        },
        "l1_fit": {
            "type": "string",
            "enum": FIT_VALUES,
            "description": "Fit/silhouette.",
        },
        "l1_length": {
            "type": "string",
            "enum": LENGTH_VALUES,
            "description": "Garment length.",
        },
        # L2 Perceptual — Universal
        **_l2_schema_properties(),
        # L3 Theory — Shared + Apparel-specific
        **_l3_shared_properties(),
        "l3_silhouette": {
            "type": "string",
            "enum": SILHOUETTE_VALUES,
            "description": "Body silhouette line classification.",
        },
        "l3_proportion_effect": {
            "type": "string",
            "enum": PROPORTION_EFFECT_VALUES,
            "description": "Visual proportion effect on wearer.",
        },
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


def _build_footwear_schema() -> dict:
    """Complete JSON schema for Footwear items (L1+L2+L3 = 21 fields)."""
    properties = {
        # L1 Product — Footwear
        "l1_material": {
            "type": "string",
            "enum": MATERIAL_VALUES_FOOTWEAR,
            "description": "Primary material category. Put details in l1_material_detail.",
        },
        "l1_closure": {
            "type": "string",
            "enum": CLOSURE_VALUES_FOOTWEAR,
            "description": "Closure type.",
        },
        "l1_design_details": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
            "description": (
                "Notable design details (at least 1 required). Include: "
                "surface finish, construction, decorative elements, "
                "graphic/print, hardware. E.g. 'perforated', 'quilted', 'logo patch'."
            ),
        },
        "l1_material_detail": {
            "type": "string",
            "description": (
                "Material composition details: blend ratio, weave, finish. "
                "E.g. 'cotton canvas, vulcanized rubber sole', "
                "'full-grain leather, Goodyear welt'. Write 'single material' if pure."
            ),
        },
        "l1_toe_shape": {
            "type": "string",
            "enum": TOE_SHAPE_VALUES,
            "description": "Toe shape.",
        },
        "l1_shaft_height": {
            "type": "string",
            "enum": SHAFT_HEIGHT_VALUES,
            "description": "Shaft height.",
        },
        "l1_heel_type": {
            "type": "string",
            "enum": HEEL_TYPE_VALUES,
            "description": "Heel type.",
        },
        "l1_sole_type": {
            "type": "string",
            "enum": SOLE_TYPE_VALUES,
            "description": "Sole type.",
        },
        # L2 Perceptual — Universal
        **_l2_schema_properties(),
        # L3 Theory — Shared + Footwear-specific
        **_l3_shared_properties(),
        "l3_foot_silhouette": {
            "type": "string",
            "enum": FOOT_SILHOUETTE_VALUES,
            "description": "Visual silhouette of the footwear.",
        },
        "l3_height_effect": {
            "type": "string",
            "enum": HEIGHT_EFFECT_VALUES,
            "description": "Effect on perceived height/proportions.",
        },
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


def _build_accessories_schema() -> dict:
    """Complete JSON schema for Accessories items (L1+L2+L3 = 21 fields)."""
    properties = {
        # L1 Product — Accessories
        "l1_material": {
            "type": "string",
            "enum": MATERIAL_VALUES_ACCESSORIES,
            "description": "Primary material category. Put details in l1_material_detail.",
        },
        "l1_closure": {
            "type": "string",
            "enum": CLOSURE_VALUES_ACCESSORIES,
            "description": "Closure type.",
        },
        "l1_design_details": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
            "description": (
                "Notable design details (at least 1 required). Include: "
                "surface finish, construction, decorative elements, "
                "graphic/print, hardware. E.g. 'beaded', 'tassel', 'monogram'."
            ),
        },
        "l1_material_detail": {
            "type": "string",
            "description": (
                "Material composition details: blend ratio, weave, finish. "
                "E.g. 'full-grain cowhide leather, gold-tone zinc alloy hardware', "
                "'925 sterling silver, rhodium plated'. Write 'single material' if pure."
            ),
        },
        "l1_form_factor": {
            "type": "string",
            "enum": FORM_FACTOR_VALUES,
            "description": "Overall form factor.",
        },
        "l1_size_scale": {
            "type": "string",
            "enum": SIZE_SCALE_VALUES,
            "description": "Size scale.",
        },
        "l1_wearing_method": {
            "type": "string",
            "enum": WEARING_METHOD_VALUES,
            "description": "How the item is worn/carried.",
        },
        "l1_primary_function": {
            "type": "string",
            "enum": PRIMARY_FUNCTION_VALUES,
            "description": "Primary functional purpose.",
        },
        # L2 Perceptual — Universal
        **_l2_schema_properties(),
        # L3 Theory — Shared + Accessories-specific
        **_l3_shared_properties(),
        "l3_visual_form": {
            "type": "string",
            "enum": VISUAL_FORM_VALUES,
            "description": "Visual form classification.",
        },
        "l3_styling_effect": {
            "type": "string",
            "enum": STYLING_EFFECT_VALUES,
            "description": "Styling effect in an outfit.",
        },
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


# Pre-built schemas
APPAREL_SCHEMA: dict = _build_apparel_schema()
FOOTWEAR_SCHEMA: dict = _build_footwear_schema()
ACCESSORIES_SCHEMA: dict = _build_accessories_schema()

SCHEMA_MAP: dict[str, dict] = {
    "Apparel": APPAREL_SCHEMA,
    "Footwear": FOOTWEAR_SCHEMA,
    "Accessories": ACCESSORIES_SCHEMA,
}


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

_COMMON_INSTRUCTIONS = """You are a fashion product analyst. Given a product's metadata, description text, and product image, extract structured attributes across three layers:

**L1 (Product):** Objective, physical attributes visible from the product itself.
**L2 (Perceptual):** Subjective impressions — style, mood, occasion, quality feel.
**L3 (Theory):** Fashion-theory-based attributes — color harmony, silhouette lines, coordination roles.

Rules:
- Use the image as the PRIMARY source for visual attributes (fit, length, silhouette, color). Use text to confirm or supplement.
- If the image is not provided, rely on text description and metadata only.
- For array fields, provide 1-3 items ordered by relevance.
- For integer scales (1-5), 3 = average/neutral.
- Be precise and consistent. Prefer specific terms from the allowed values.

Cross-Attribute Consistency Rules (MUST follow):
- visual_weight = FORM and VOLUME (3D structure), NOT color darkness or pattern complexity.

Per-Item Specificity (IMPORTANT):
- Do NOT assign generic defaults based on product_type alone.
- Two T-shirts with different designs MUST get different attribute values.
- Examine the IMAGE carefully: prints, graphics, embellishments, unusual cuts, hardware.
- style_mood: Base on THIS item's visual character, not category stereotypes.
- Any visible pattern/graphic/embellishment MUST appear in design_details.
- perceived_quality: Differentiate using material and construction cues — most items should NOT default to 3."""

SYSTEM_PROMPT_APPAREL: str = f"""{_COMMON_INSTRUCTIONS}

You are analyzing an **APPAREL** item (clothing worn on the body).

Category-specific L1 attributes:
- neckline: The neckline style. Use 'N/A' for bottoms or items without a neckline.
- sleeve_type: The sleeve style. Use 'N/A' for sleeveless items or bottoms.
- fit: How the garment fits the body.
- length: Where the garment ends on the body. Use 'N/A' if not applicable.

Category-specific L3 attributes:
- silhouette: The body silhouette line (A-line, H-line, I-line, etc.).
- proportion_effect: How this garment affects the wearer's visual proportions.

Example output for a "Black slim-fit crew neck T-shirt, cotton":
{{
  "l1_material": "Cotton",
  "l1_closure": "Pullover",
  "l1_design_details": ["ribbed neckline"],
  "l1_material_detail": "100% cotton, jersey knit",
  "l1_neckline": "Crew",
  "l1_sleeve_type": "Short",
  "l1_fit": "Slim",
  "l1_length": "Hip",
  "l2_style_mood": ["Casual", "Minimalist"],
  "l2_occasion": ["Everyday", "Active"],
  "l2_perceived_quality": 3,
  "l2_trendiness": "Classic",
  "l2_season_fit": "All-season",
  "l2_target_impression": "effortless everyday essential",
  "l2_versatility": 5,
  "l3_color_harmony": "Monochromatic",
  "l3_coordination_role": "Basic",
  "l3_visual_weight": 2,
  "l3_style_lineage": ["Scandinavian Minimalism"],
  "l3_silhouette": "I-line",
  "l3_proportion_effect": "Streamlining"
}}

Contrast: "Red oversized wool coat" → coordination_role="Statement", visual_weight=4, silhouette="O-line"."""

SYSTEM_PROMPT_FOOTWEAR: str = f"""{_COMMON_INSTRUCTIONS}

You are analyzing a **FOOTWEAR** item (shoes, boots, sandals, socks, tights).

Category-specific L1 attributes:
- toe_shape: The shape of the toe box. Use 'N/A' for socks, tights, or open footwear without a defined toe box.
- shaft_height: How high the shoe/boot shaft extends. Use 'N/A' for open shoes.
- heel_type: The heel style. Use 'N/A' for flat shoes without distinct heel.
- sole_type: The sole construction.

Category-specific L3 attributes:
- foot_silhouette: The overall visual silhouette of the footwear.
- height_effect: How this footwear affects perceived height/proportions.

Example output for a "White canvas low-top sneaker, rubber sole, lace-up":
{{
  "l1_material": "Canvas",
  "l1_closure": "Lace-up",
  "l1_design_details": ["rubber toe cap", "metal eyelets"],
  "l1_material_detail": "cotton canvas, vulcanized rubber sole",
  "l1_toe_shape": "Round",
  "l1_shaft_height": "Low-top",
  "l1_heel_type": "Flat",
  "l1_sole_type": "Rubber",
  "l2_style_mood": ["Casual", "Sporty"],
  "l2_occasion": ["Everyday", "Active"],
  "l2_perceived_quality": 3,
  "l2_trendiness": "Classic",
  "l2_season_fit": "All-season",
  "l2_target_impression": "clean casual versatility",
  "l2_versatility": 5,
  "l3_color_harmony": "Monochromatic",
  "l3_coordination_role": "Basic",
  "l3_visual_weight": 2,
  "l3_style_lineage": ["Americana Prep"],
  "l3_foot_silhouette": "Streamlined",
  "l3_height_effect": "Grounding"
}}

Contrast: "Chunky platform boots" → coordination_role="Statement", visual_weight=4, foot_silhouette="Chunky"."""

SYSTEM_PROMPT_ACCESSORIES: str = f"""{_COMMON_INSTRUCTIONS}

You are analyzing an **ACCESSORY** item (bags, jewelry, scarves, hats, belts, cosmetics, home items, stationery).

Category-specific L1 attributes:
- form_factor: The overall size and structure.
- size_scale: The relative size.
- wearing_method: How the item is worn or carried. Use 'N/A' for non-wearable items.
- primary_function: The main functional purpose.

Category-specific L3 attributes:
- visual_form: The visual form classification.
- styling_effect: The styling effect when combined with an outfit.

Example output for a "Black leather crossbody bag with gold hardware":
{{
  "l1_material": "Leather",
  "l1_closure": "Magnetic",
  "l1_design_details": ["gold hardware", "adjustable strap"],
  "l1_material_detail": "full-grain cowhide leather, gold-tone zinc alloy hardware",
  "l1_form_factor": "Small",
  "l1_size_scale": "Small",
  "l1_wearing_method": "Crossbody",
  "l1_primary_function": "Storage",
  "l2_style_mood": ["Classic", "Minimalist"],
  "l2_occasion": ["Everyday", "Work"],
  "l2_perceived_quality": 4,
  "l2_trendiness": "Classic",
  "l2_season_fit": "All-season",
  "l2_target_impression": "polished everyday companion",
  "l2_versatility": 4,
  "l3_color_harmony": "Monochromatic",
  "l3_coordination_role": "Finishing",
  "l3_visual_weight": 2,
  "l3_style_lineage": ["French Chic"],
  "l3_visual_form": "Structured",
  "l3_styling_effect": "Cohesion"
}}

Contrast: "Oversized woven tote" → coordination_role="Statement", visual_weight=4, visual_form="Structured".

For design_details, Accessories should capture:
- Surface: textured, smooth, polished, matte, glossy
- Construction: stitching, rivets, hardware (gold/silver), chain type
- Decoration: embossing, logo, pattern, stones, beads
- If truly minimal (plain metal ring): ["polished finish"] or similar."""

SYSTEM_PROMPT_MAP: dict[str, str] = {
    "Apparel": SYSTEM_PROMPT_APPAREL,
    "Footwear": SYSTEM_PROMPT_FOOTWEAR,
    "Accessories": SYSTEM_PROMPT_ACCESSORIES,
}


# ---------------------------------------------------------------------------
# Category-Specific Field Mapping (slot4~7)
# ---------------------------------------------------------------------------

CATEGORY_SPECIFIC_L1_FIELDS: dict[str, list[str]] = {
    "Apparel": ["l1_neckline", "l1_sleeve_type", "l1_fit", "l1_length"],
    "Footwear": ["l1_toe_shape", "l1_shaft_height", "l1_heel_type", "l1_sole_type"],
    "Accessories": ["l1_form_factor", "l1_size_scale", "l1_wearing_method", "l1_primary_function"],
}

CATEGORY_SPECIFIC_L3_FIELDS: dict[str, list[str]] = {
    "Apparel": ["l3_silhouette", "l3_proportion_effect"],
    "Footwear": ["l3_foot_silhouette", "l3_height_effect"],
    "Accessories": ["l3_visual_form", "l3_styling_effect"],
}

# Parquet canonical slot names
L1_SLOT_NAMES = ["l1_slot4", "l1_slot5", "l1_slot6", "l1_slot7"]
L3_SLOT_NAMES = ["l3_slot6", "l3_slot7"]


def map_to_canonical_slots(knowledge: dict, super_category: str) -> dict:
    """Map category-specific field names to canonical slot names for Parquet storage.

    E.g., Apparel's l1_neckline → l1_slot4, l1_sleeve_type → l1_slot5, etc.
    """
    result = dict(knowledge)
    l1_fields = CATEGORY_SPECIFIC_L1_FIELDS[super_category]
    for slot_name, field_name in zip(L1_SLOT_NAMES, l1_fields):
        result[slot_name] = result.pop(field_name, None)
    l3_fields = CATEGORY_SPECIFIC_L3_FIELDS[super_category]
    for slot_name, field_name in zip(L3_SLOT_NAMES, l3_fields):
        result[slot_name] = result.pop(field_name, None)
    return result


# ---------------------------------------------------------------------------
# Message Building
# ---------------------------------------------------------------------------


def build_user_message(
    article: dict,
    detail_desc: str,
    image_base64: str | None,
) -> list[dict]:
    """Build OpenAI user message content (text + optional image).

    Args:
        article: Metadata dict (product_type_name, colour_group_name, etc.).
        detail_desc: Product description text.
        image_base64: Base64-encoded JPEG image, or None if unavailable.

    Returns:
        List of content blocks for OpenAI messages format.
    """
    # Build text portion from metadata
    meta_parts = []
    for field in [
        "product_type_name",
        "product_group_name",
        "colour_group_name",
        "perceived_colour_value_name",
        "perceived_colour_master_name",
        "graphical_appearance_name",
        "department_name",
        "index_group_name",
        "section_name",
    ]:
        val = article.get(field)
        if val and str(val).strip() and str(val) != "nan":
            meta_parts.append(f"{field}: {val}")

    metadata_text = "\n".join(meta_parts)

    # Build text content
    text_content = f"Product metadata:\n{metadata_text}"
    if detail_desc and str(detail_desc).strip() and str(detail_desc) != "nan":
        text_content += f"\n\nProduct description:\n{detail_desc}"
    else:
        text_content += "\n\n(No text description available. Rely on image and metadata.)"

    text_content += (
        f"\n\nAnalyze THIS specific {article.get('product_type_name', 'item')}'s "
        "unique visual details. Do not assign generic attributes."
    )

    content: list[dict] = [{"type": "input_text", "text": text_content}]

    # Add image if available (Responses API format)
    if image_base64:
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{image_base64}",
                "detail": "low",
            }
        )

    return content


def get_prompt_and_schema(super_category: str) -> tuple[str, dict]:
    """Return (system_prompt, json_schema) for a Super-Category."""
    if super_category not in SYSTEM_PROMPT_MAP:
        raise ValueError(
            f"Unknown super_category: {super_category}. "
            f"Expected one of: {list(SYSTEM_PROMPT_MAP.keys())}"
        )
    return SYSTEM_PROMPT_MAP[super_category], SCHEMA_MAP[super_category]
