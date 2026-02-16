# =============================================================================
# OPTIMIZED HOLE DETECTION CONFIGURATION
# =============================================================================
# Goal: Detect HOLES reliably while filtering wrinkles, folds, and hands
# Strategy: Start permissive, use smart validation to reject false positives
# =============================================================================

# -----------------------------------------------------------------------------
# MODEL & SERVER
# -----------------------------------------------------------------------------
MODEL_PATH = "notebook/best.pt"
NODE_SERVER = "http://localhost:3000"

# -----------------------------------------------------------------------------
# DETECTION THRESHOLDS - OPTIMIZED FOR SPEED
# -----------------------------------------------------------------------------
CONF_THRESHOLD = 0.50           # Lower for faster detection (was 0.60)
IOU_THRESHOLD = 0.45            # Standard overlap threshold

# -----------------------------------------------------------------------------
# SIZE CONSTRAINTS - REASONABLE RANGE
# -----------------------------------------------------------------------------
MIN_DEFECT_AREA = 800           # Minimum hole size (smaller than before)
MAX_DEFECT_AREA = 50000         # Maximum (reject very large detections)
MIN_WIDTH = 10                  # Minimum width
MIN_HEIGHT = 10                 # Minimum height

# -----------------------------------------------------------------------------
# SHAPE ANALYSIS - FLEXIBLE BUT EFFECTIVE
# -----------------------------------------------------------------------------
MIN_ASPECT_RATIO = 0.20         # Very flexible (holes can be irregular)
CIRCULARITY_THRESHOLD = 0.25    # Low threshold (holes aren't perfect circles)
MIN_CONTOUR_POINTS = 5          # Minimum shape complexity

# -----------------------------------------------------------------------------
# DARKNESS ANALYSIS - KEY HOLE CHARACTERISTIC
# -----------------------------------------------------------------------------
DARKNESS_THRESHOLD = 120        # Adjusted for bright fabric
DARKNESS_RATIO_MIN = 0.25       # At least 25% must be darker
CENTER_DARKNESS_THRESHOLD = 110 # Center should be darker

# -----------------------------------------------------------------------------
# TEXTURE ANALYSIS - SEPARATE HOLES FROM WRINKLES
# -----------------------------------------------------------------------------
TEXTURE_STD_MIN = 5             # Holes have some texture variation
TEXTURE_STD_MAX = 40            # Wrinkles have very high variation

# -----------------------------------------------------------------------------
# EDGE ANALYSIS - HOLES HAVE DEFINED EDGES
# -----------------------------------------------------------------------------
EDGE_DENSITY_MIN = 0.01         # Minimum edge presence
EDGE_DENSITY_MAX = 0.40         # Maximum (too many = noise)
EDGE_STRENGTH_THRESHOLD = 50    # Minimum edge strength

# -----------------------------------------------------------------------------
# COLOR ANALYSIS - REJECT HANDS AND COLORED OBJECTS
# -----------------------------------------------------------------------------
MAX_SATURATION = 120            # Reject highly saturated areas
SKIN_HUE_RANGE = (0, 25)        # Skin tone hue range
SKIN_SATURATION_RANGE = (30, 180)  # Skin tone saturation range

# -----------------------------------------------------------------------------
# CONTEXT ANALYSIS - SURROUNDING AREA
# -----------------------------------------------------------------------------
MAX_SURROUNDING_BRIGHTNESS = 210    # Allow bright backgrounds
MIN_CONTRAST_WITH_SURROUND = 8      # Minimum contrast needed

# -----------------------------------------------------------------------------
# TRACKING & NOTIFICATION
# -----------------------------------------------------------------------------
DEFECT_COOLDOWN = 0.1           # Faster notification (was 0.3)
DEFECT_DISTANCE_THRESHOLD = 50  # Tracking distance (pixels)

# -----------------------------------------------------------------------------
# VALIDATION STRATEGY
# -----------------------------------------------------------------------------
# We use a weighted scoring system instead of requiring ALL checks to pass
# This allows for more flexible detection while still filtering false positives

VALIDATION_WEIGHTS = {
    'darkness': 3.0,        # Most important - holes are darker
    'texture': 2.0,         # Important - wrinkles have high texture
    'edges': 1.5,           # Somewhat important
    'circularity': 1.0,     # Less important - holes can be irregular
    'color': 2.5,           # Important - reject hands
    'context': 1.0          # Supporting evidence
}

MIN_VALIDATION_SCORE = 5.0  # Lower for faster detection (was 6.0)

# =============================================================================
# OPTIMIZATION NOTES
# =============================================================================
"""
KEY CHANGES FROM PREVIOUS VERSION:
1. Lower confidence threshold (0.60 vs 0.80) - catches more detections
2. Weighted scoring instead of all-or-nothing validation
3. More flexible shape requirements
4. Better texture analysis to separate holes from wrinkles
5. Faster tracking and notification

WHAT THIS CATCHES:
- Real holes in fabric (various sizes and shapes)
- Tears and punctures
- Missing fabric areas

WHAT THIS REJECTS:
- Wrinkles and folds (high texture variance)
- Hands and skin (color/saturation analysis)
- Shadows (context analysis)
- Fabric patterns (edge and texture analysis)
- Very large detections (size limits)

TUNING TIPS:
- If missing holes: Decrease CONF_THRESHOLD to 0.55
- If too many wrinkles: Increase TEXTURE_STD_MAX to reject high variance
- If detecting hands: Increase color analysis weight
- If missing small holes: Decrease MIN_DEFECT_AREA
"""