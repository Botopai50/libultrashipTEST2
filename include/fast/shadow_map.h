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
// Slope-scaled bias, handed to the rasterizer: a multiple of the polygon's own depth gradient across a
// texel. Being relative to the gradient is exactly right -- it is nearly nothing on a surface facing the
// light and large on one edge-on to it, which is where depth runs away across a texel and acne appears.
// Being relative to the TEXEL is the part that misbehaved, since a texel of the far cascade is several
// world units, and that is now handled per cascade by the ceiling below.
//
// 4.0, back where it started. It was halved to 2.0 to fight peter panning before the per-cascade ceiling
// existed, which was the wrong instrument: halving it globally took the bias away from the near and middle
// cascades, where the panning was never the problem and the acne is. On a castle wall -- a vertical surface
// with the light coming from above, so nearly edge-on to it -- that showed up as horizontal bars, because
// the lines of constant depth in the map run horizontally across such a wall.
#define SHADOW_MAP_DEFAULT_SLOPE_BIAS 4.0f

// Ceiling on what that slope term may displace a receiver by, in WORLD units.
//
// The rasterizer takes one slope value per state, not per draw, so this cannot be capped inside the shader
// the way the normal offset is -- the backend builds a separate rasterizer state per cascade instead, each
// with the slope reduced to whatever keeps its own texel under this ceiling.
//
// 32.0, which at the default cascade ladder and resolution does not bind on ANY cascade: the far one would
// need a texel over eight world units to reach it, and it sits at about six. So this is a guard rail rather
// than a working part -- it exists for configurations with much coarser texels, a raised Split3 or a
// lowered Resolution, where the term would otherwise run away.
//
// It was 3.0, then 6.0, and both were too tight. The mistake in each was pricing this term as though it
// cost everywhere, when it scales with the polygon's own gradient: near zero on a surface facing the light,
// which is where a shadow's contact point is read closely, and large only on grazing surfaces, where the
// contact is at a shallow angle and displacement along the ray barely shows. Capping it there bought very
// little panning and cost the acne protection the far cascade needs.
#define SHADOW_MAP_MAX_SLOPE_BIAS_WORLD 32.0f

// How far along the surface normal the receiver is nudged before the comparison, in cascade texels.
// This is the term that actually removes the striped self-shadowing (acne) on grazing and curved surfaces,
// where a depth-only bias cannot: it moves the sample sideways off the shadow-casting surface itself rather
// than sliding it along the light ray, still inside the same polygon.
// Scaling with the texel is what makes one value work across cascades -- acne appears at the scale of the
// map's own resolution, so the push has to be measured in the same unit.
// The shader scales this by the sine of the angle between the surface and the light, so it is the offset at
// a fully grazing angle, not everywhere. That is where acne lives; a surface square to the light needs
// nothing and now pays nothing.
// 3.0 rather than 2.0: the offset costs nothing on the surfaces where its cost was visible, so it can afford
// to be larger on the ones where it does the work. It remains the primary defence against self-shadowing,
// having replaced both the front-face culling that caused peter panning and half of the constant bias that
// did the same -- and now most of the slope bias too, which had to be capped per cascade to stop it
// detaching distant shadows.
#define SHADOW_MAP_DEFAULT_NORMAL_OFFSET 3.0f

// Ceiling on that push, in WORLD units, applied per cascade before the shader sees it.
//
// Measuring the offset in texels is right while a texel is small, but a texel of the far cascade is several
// world units, so two of them is nearly twelve -- and the shader moves the receiver's sample that far
// towards the light before comparing. The shadow lifts off whatever cast it, by more the further away the
// receiver is, because the cascade it lands in is coarser. That is peter panning that grows with distance.
//
// 3.0 is under a tenth of a character's height, so the detachment stays below what the eye reads as a gap,
// and it only starts binding partway through the third cascade -- the near cascades keep tracking their
// texel exactly as before.
#define SHADOW_MAP_MAX_NORMAL_OFFSET_WORLD 3.0f

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
// Four bilinear taps at +/- this radius, each already spanning a 2x2 texel quad, so the kernel covers
// 2*(1+radius) texels across. At 1.0 the quads sit edge to edge and cover 4x4; above that they separate and
// leave texels sampled by nothing, which reads on screen as a grid, so the shader clamps there.
//
// 0.5, and 0.5 is the floor. It was tried at 0.25 on the reasoning that bilinear taps cannot stair-step --
// that reasoning was wrong. A bilinear tap removes the hardness WITHIN a texel; it does nothing about the
// staircase BETWEEN texels, because the shadow edge is quantised to the texel grid either way and a
// one-texel ramp only rounds each step's corner. Hiding a staircase needs a kernel spanning several texels,
// which is what this radius buys and what 0.25 gave up.
//
// So sharper distant shadows are not available from the filter. That is a texel problem, and both cures are
// priced: halve Graphics.ShadowMap.Split3 to halve the far cascade's texel at the cost of range, or double
// Graphics.ShadowMap.Resolution to halve every texel at the cost of four times the memory.
#define SHADOW_MAP_DEFAULT_FILTER_WIDTH 0.5f

// Smallest caster the actor layer will accept, as the largest side of its world-space bounding box.
// Ground clutter -- grass tufts, flowers, small debris -- is armed as a caster like anything else, and at
// that scale a shadow is a few texels of smudge that reads as dirt rather than as a shadow, while still
// costing a full re-rasterisation in every cascade.
//
// OFF by default, because a threshold is a cliff and a caster sitting on it flickers. 40 units was chosen
// against adult Link and is roughly CHILD Link's whole height, so his shadow crossed the threshold as the
// walk cycle bobbed him and blinked once per step. Any value has a band of casters it does this to, and a
// character blinking is far worse than clutter casting -- so the game names the clutter it does not want
// casting by actor id (which is exact) and this stays available for tuning rather than guessing on
// everyone's behalf.
//
// Raise it if ground clutter still casts. If a character starts flickering, the value is too close to its
// height: halve it.
#define SHADOW_MAP_DEFAULT_MIN_CASTER_SIZE 0.0f

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

// How far around the camera an actor is drawn purely so it can cast, in world units, in EVERY direction.
//
// The game's own culling is a screen-space test, so it drops anything behind the camera outright -- but the
// light comes from the sky, not from the eye, and a caster behind the camera casts into the view perfectly
// well. Widening the screen-space margin cannot reach these: behind the camera the projected w is negative
// and the test degenerates. Only a radial test is orientation-independent.
//
// 800 rather than the full caster reach: this draws actors nothing can see, so it is paid in draw calls
// across the whole sphere. 800 covers the space immediately around and behind the camera, which is where a
// caster is close enough for its shadow to land inside the view.
#define SHADOW_MAP_DEFAULT_CASTER_DRAW_RADIUS 800.0f

// How far the key light may swing before the cascades are rebuilt around the new direction, as the cosine
// of the angle (0.9994 is about two degrees).
//
// The cascade centre is snapped to whole texels along the LIGHT's own axes, which is what stops shadow
// edges shimmering as the camera moves. That only works if those axes hold still: the game's environment
// light turns continuously with the time of day, and a basis that turns with it is a grid that turns with
// it, so the snapping is measuring against a ruler that keeps rotating and every edge trembles.
//
// Held with hysteresis rather than quantized, for the same reason the cascade radius is: a quantized
// direction flips between two neighbouring steps whenever it sits near a boundary. The cost is that the
// whole shadow grid re-aligns in one step when the threshold is crossed, so the threshold wants to stay
// small enough that the step is not visible.
#define SHADOW_MAP_LIGHT_DIR_HYSTERESIS_COS 0.9994f

// Strength of the shadow where it is fully occluded (0 = invisible, 1 = black).
#define SHADOW_MAP_DEFAULT_STRENGTH 0.5f

#endif // FAST_SHADOW_MAP_H
