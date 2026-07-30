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

// How many are actually built unless the application says otherwise. All of them: the split ladder below is
// absolute distances and the count selects how much of it is used, so lowering the count is not a quality
// knob -- it shortens the shadowed range and the caster capture reach with it.
#define SHADOW_MAP_DEFAULT_CASCADES SHADOW_MAP_MAX_CASCADES

// Casters are split into two layers, because the four interaction rules cannot be satisfied by one map:
// scenery must be shadowed by characters, while characters must NOT be shadowed by characters. The same
// map cannot both contain and not contain the actors, so there are two.
//   World layer  -- scenery casters only. Sampled by everything.
//   Actor layer  -- character casters only. Sampled by scenery, never by characters.
// Both live in ONE texture array with twice the slices rather than two arrays: the slice is a texture
// coordinate, so selecting a layer is an index offset and the shader needs only one sampler and one set of
// sampling code. Layer L, cascade C is slice L*cascadeCount + C.
#define SHADOW_MAP_LAYERS 2
#define SHADOW_MAP_LAYER_WORLD 0
#define SHADOW_MAP_LAYER_ACTORS 1
#define SHADOW_MAP_MAX_SLICES (SHADOW_MAP_MAX_CASCADES * SHADOW_MAP_LAYERS)

// Per-cascade square resolution bounds. 4096 is the largest the near cascade is ever asked for, and
// anything under 256 produces texels so large that the bias needed to hide the acne swallows the
// shadow itself.
#define SHADOW_MAP_MIN_RESOLUTION 256
#define SHADOW_MAP_MAX_RESOLUTION 4096
#define SHADOW_MAP_DEFAULT_RESOLUTION 2048

// Default split distances (world units from the camera) for the four cascades. These bound the far
// plane of each cascade's ortho projection; the near plane of cascade N is the far plane of N-1.
//
// Note that the last ACTIVE split -- split[cascadeCount - 1], not split 3 -- is also the distance out to
// which the application captures casters at all. These values are the range; the cascade count selects how
// much of it is used, so lowering the count shortens the shadowed range with it.
#define SHADOW_MAP_DEFAULT_SPLIT_0 150.0f
#define SHADOW_MAP_DEFAULT_SPLIT_1 350.0f
#define SHADOW_MAP_DEFAULT_SPLIT_2 1500.0f
#define SHADOW_MAP_DEFAULT_SPLIT_3 6000.0f

// Fraction of a cascade's range over which it cross-fades into the next one. The shader samples both
// maps across this band and blends with smoothstep, which is what keeps the resolution change from
// showing up as a hard line sweeping across the ground as the camera moves ("cascade popping").
// Expressed as a fraction so the band scales with each cascade's size.
#define SHADOW_MAP_DEFAULT_BLEND_FRACTION 0.1f

// Depth bias for the shadow comparison, in WORLD UNITS and applied in the shader, not handed to the rasterizer.
//
// The rasterizer's own constant bias counts smallest representable depth increments, and each cascade
// covers a different depth range -- so one setting means wildly different distances per cascade. At 150
// increments the near cascade got about 1.3 world units and the far one about 70, which detached distant
// shadows from their casters and, worse, placed the SAME shadow in a different spot in each cascade: at a
// cascade transition you could see both copies at once, metres apart.
//
// Expressed in world units and divided by each cascade's own depth range at upload time, one value means
// the same physical offset everywhere. The slope term stays with the rasterizer, where being relative to
// the polygon's own gradient is exactly what it should be.
// 1.0 rather than 2.0: the normal-offset term below now does real work on the room mesh (the shader
// recovers a face normal from screen derivatives where no vertex normal exists), so the constant term no
// longer has to carry the whole load, and every unit of it is peter panning.
#define SHADOW_MAP_DEFAULT_DEPTH_BIAS_WORLD 1.0f
#define SHADOW_MAP_DEFAULT_SLOPE_BIAS 4.0f

// How far along the surface normal the receiver is nudged before the comparison, in cascade texels.
// This is the term that actually removes the striped self-shadowing (acne) on grazing and curved surfaces,
// where a depth-only bias cannot: it moves the sample sideways off the shadow-casting surface itself rather
// than sliding it along the light ray, still inside the same polygon.
// Scaling with the texel is what makes one value work across cascades -- acne appears at the scale of the
// map's own resolution, so the push has to be measured in the same unit.
// 2.0 rather than 1.0: this term is now the primary defence against self-shadowing, replacing both the
// front-face culling that caused peter panning and half of the constant bias that did the same.
#define SHADOW_MAP_DEFAULT_NORMAL_OFFSET 2.0f

// Floor on how low the key light may sit before the cascades are built from it, as the sine of its angle
// above the horizon (0 = the horizon itself, 1 = straight overhead). A light near the horizon stretches
// every shadow towards infinity, which reads as wrong long before it is geometrically wrong -- and it also
// wastes the cascade, since the projection has to cover a footprint far longer than the scene it is
// shading. The light's compass bearing is preserved; only its height is lifted.
// 0.5 is 30 degrees, which caps a shadow at about 1.7x the caster's height.
#define SHADOW_MAP_DEFAULT_MIN_ELEVATION 0.5f

// Radius of the PCF kernel, in cascade texels. The filter is sixteen fetches whatever this is; the value
// only says how far apart they sit, so it trades edge softness against fetch coverage at no extra cost.
//
// 0.5 rather than 1.0: at a full texel the four bilinear taps put their 2x2 quads edge to edge and cover a
// 4x4 neighbourhood, which is what finally killed the stair-stepping -- but 4x4 texels of the third cascade
// is several world units of penumbra, and shadows came out looking smeared rather than soft. At half a
// texel the quads overlap into roughly 3x3, which still filters (the failure mode to avoid is taps landing
// inside ONE texel, which is what a POINT-sampled quincunx at this radius would do -- these are bilinear,
// so every tap is already an interpolation over four texels).
#define SHADOW_MAP_DEFAULT_FILTER_WIDTH 0.5f

// Smallest caster the actor layer will accept, as the largest side of its world-space bounding box.
// Ground clutter -- grass tufts, flowers, small debris -- is armed as a caster like anything else, but at
// this scale a shadow is a few texels wide and reads as a smudge of dirt rather than as a shadow, while
// still costing a full re-rasterisation in every cascade. 40 units sits between that clutter and the props
// worth casting (Link is roughly 60 tall).
#define SHADOW_MAP_DEFAULT_MIN_CASTER_SIZE 40.0f

// Alpha below which a textured caster's pixel is punched out of the depth map entirely.
//
// Foliage in this game is alpha-cutout billboards -- a quad with a leaf texture. A depth-only pass with no
// pixel shader records the whole quad, so a tree casts a rectangle and a flowering plant casts a solid
// slab. The alpha caster path samples the material's own texture and clips, which is the only way the
// shadow can take the shape of the leaves rather than of the polygon holding them.
//
// 0.5 rather than the main pass's cutout: a shadow silhouette wants the solid core of the leaf, not its
// antialiased fringe, and a lower threshold makes every soft edge in the texture cast a full-strength
// shadow that reads as grime.
#define SHADOW_MAP_ALPHA_CUTOUT 0.5f

// Strength of the shadow where it is fully occluded (0 = invisible, 1 = black).
#define SHADOW_MAP_DEFAULT_STRENGTH 0.5f

#endif // FAST_SHADOW_MAP_H
