#ifndef FAST_SHADOW_MAP_H
#define FAST_SHADOW_MAP_H

// Shared limits and default parameters for the cascaded shadow-map effect.
//
// The effect renders the frame's shadow casters depth-only from the key light's point of view into a
// small array of depth maps ("cascades"), each covering a progressively larger slice of the view, and
// the main pass compares each shaded pixel against them. This is the same policy split the toon
// lighting uses: the framework owns the technique, the application owns the tuning and pushes it in via
// GfxRenderingAPI, so no app-specific CVar keys live here.
//
// Only the Direct3D 11 backend implements the depth pass. Everywhere else
// GfxRenderingAPI::SupportsShadowMap() answers false and the application is expected to fall back to
// whatever shadow system it used before -- selecting this mode must never leave a scene with no shadows.

// Cascade count is a compile-time bound because the shader indexes a fixed-size array of matrices.
// Four matches the layer split the design calls for (ultra near / world near / world mid / world far);
// the application may configure fewer at runtime, never more.
#define SHADOW_MAP_MAX_CASCADES 4

// Per-cascade square resolution bounds. 4096 is the largest the near cascade is ever asked for, and
// anything under 256 produces texels so large that the bias needed to hide the acne swallows the
// shadow itself.
#define SHADOW_MAP_MIN_RESOLUTION 256
#define SHADOW_MAP_MAX_RESOLUTION 4096
#define SHADOW_MAP_DEFAULT_RESOLUTION 2048

// Default split distances (world units from the camera) for the four cascades. These bound the far
// plane of each cascade's ortho projection; the near plane of cascade N is the far plane of N-1.
#define SHADOW_MAP_DEFAULT_SPLIT_0 150.0f
#define SHADOW_MAP_DEFAULT_SPLIT_1 350.0f
#define SHADOW_MAP_DEFAULT_SPLIT_2 1500.0f
#define SHADOW_MAP_DEFAULT_SPLIT_3 6000.0f

// Fraction of a cascade's range over which it cross-fades into the next one. The shader samples both
// maps across this band and blends with smoothstep, which is what keeps the resolution change from
// showing up as a hard line sweeping across the ground as the camera moves ("cascade popping").
// Expressed as a fraction so the band scales with each cascade's size.
#define SHADOW_MAP_DEFAULT_BLEND_FRACTION 0.1f

// Depth bias applied while rendering the depth maps, in the rasterizer's own units: the constant term
// counts smallest-representable depth increments of the map's format (D16 here), and the slope-scaled
// term multiplies the polygon's depth gradient. Constant bias alone would have to be large enough for
// the steepest surface in the scene, which detaches contact shadows everywhere else ("peter panning"),
// so the slope-scaled term carries most of the load and the constant one stays small.
// The constant term is an integer because that is what the rasterizer takes for a UNORM depth format.
//
// Scale matters here and is easy to get wrong: the unit is one smallest representable depth increment, not
// one world unit and not one texel. A D16 map spanning a few thousand world units resolves roughly a
// hundredth of a unit per increment, so a value like 2 is worth a few hundredths of a unit -- far too small
// to lift a surface off its own depth values. That was the original mistake, and it showed up as terrain
// shadowing itself across the whole cascade footprint: one enormous dark quad on the ground with straight
// edges, which were the cascade's own borders.
#define SHADOW_MAP_DEFAULT_CONSTANT_BIAS 150
#define SHADOW_MAP_DEFAULT_SLOPE_BIAS 4.0f

// How far along the surface normal the receiver is nudged before the comparison, in cascade texels.
// This is the term that actually removes the striped self-shadowing (acne) on curved surfaces, where a
// depth-only bias cannot: it moves the sample sideways off the shadow-casting surface itself.
#define SHADOW_MAP_DEFAULT_NORMAL_OFFSET 1.0f

// Strength of the shadow where it is fully occluded (0 = invisible, 1 = black).
#define SHADOW_MAP_DEFAULT_STRENGTH 0.5f

#endif // FAST_SHADOW_MAP_H
