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
// Three matches the layer split the design calls for (ultra near / world near / world far); the application
// may configure fewer at runtime, never more.
//
// Three rather than four, and it is a bound rather than only a default because the fourth was measured and
// found not to earn its place. The depth pass costs per SLICE -- about 0.42 ms each at 4096 -- and a fourth
// cascade is two of them, for a band whose map is spread over the frustum's lateral width at six thousand
// units and is coarse no matter where it starts. It also costs on the receiver side, where the shader
// selects a cascade's constants with a chain of comparisons and every arm is registers moved for every
// shaded pixel. Bounding it here retires both.
#define SHADOW_MAP_MAX_CASCADES 3

// How many are actually built unless the application says otherwise.
//
// Three, not four, and the ladder below is drawn to match so the RANGE is unchanged by that -- the count
// selects how much of an absolute ladder is used, so shortening it without redrawing the ladder would simply
// end shadows sooner. Redrawn, it is a straight trade of resolution in the middle distance for two fewer
// slices, and slices are what the depth pass costs: measured at roughly 0.42 ms each at 4096, against a pass
// that was 2.3 ms in a field.
//
// What it costs is the band beyond the middle split, which now falls in one cascade instead of two and comes
// out about twice as coarse. What it does not cost is anything close to the camera: the first two bands are
// untouched, and they are where a shadow's edge is actually read.
#define SHADOW_MAP_DEFAULT_CASCADES 3

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
// How many cascades the ACTOR layer gets, which is fewer than the world layer's.
//
// That layer costs a full slice per cascade and is redrawn every frame no matter what: its contents are
// animating characters, so its content key changes every frame and no amount of reuse machinery can park it.
// At 4096 a slice is about 0.42 ms of clear, and the far one was buying character shadows on ground the
// player is looking at from over a thousand units away, where a character is a few pixels tall.
//
// What it costs is exactly that: past the second band, characters stop casting onto scenery. They still
// RECEIVE the world layer's shadows at every distance -- this shortens the casting, not the shading.
#define SHADOW_MAP_ACTOR_CASCADES 2

// Slices are laid out world-layer-first: world cascade C is slice C, actor cascade C is slice
// cascadeCount + C. The actor half is the shorter one, so the total is not a simple product.
#define SHADOW_MAP_ACTOR_CASCADES_FOR(count) ((count) < SHADOW_MAP_ACTOR_CASCADES ? (count) : SHADOW_MAP_ACTOR_CASCADES)
#define SHADOW_MAP_SLICES_FOR(count) ((count) + SHADOW_MAP_ACTOR_CASCADES_FOR(count))
#define SHADOW_MAP_MAX_SLICES SHADOW_MAP_SLICES_FOR(SHADOW_MAP_MAX_CASCADES)

// SOH [Enhancement] Content key meaning "nothing will be drawn into this slice at all".
//
// A slice with no casters in it holds the clear value everywhere, and a map of nothing but the clear value
// reads the same however it is projected -- every receiver compares as lit against the far plane wherever it
// lands. So an empty slice, unlike a drawn one, may be kept across a MATRIX change as well as a content one,
// and the backend exempts this key from that comparison. The character layer is empty in most cascades most
// of the time (the characters stand in one place; the cascades cover the whole view), and without this it
// paid a full-resolution clear and a pipeline setup in each of them every frame.
//
// Reserved, so a real key can never collide with it: ShadowMapLayerContentKey moves a hash that lands here
// off it. Losing one value out of 2^64 costs nothing.
#define SHADOW_MAP_EMPTY_CONTENT_KEY 0ull

// SOH [Enhancement] How much larger than the sphere it was fitted to each cascade is built, as a fraction
// of its radius. This is what lets a cascade stay PARKED across frames.
//
// A slice is redrawn only when its matrix changes, and the matrix changes when the cascade moves. Holding
// it still is exact -- a cascade that still contains the fitted sphere shows every caster and every receiver
// the fitted one would -- but it can only be held while there is room to hold it, and the room is exactly
// this margin. Without it the only slack was whatever the radius quantisation happened to leave, which
// measured out at 0 to 28 world units against a camera that moves ten per frame: the cascades re-parked
// every frame and nothing was ever reused.
//
// The cost is that the radius sets the texel size, so this widens the penumbra by the same fraction. That
// is a real change and it is bounded rather than assumed: the kernel is four texels, a texel is 2R/
// resolution, so the penumbra grows by 4 * 2R * margin / resolution world units -- while a screen pixel at
// the nearest distance the cascade covers grows in the same proportion, because both scale with distance.
// At 6%, 1080p and a 60 degree field of view that comes out at 0.55, 0.55 and 0.26 of a SCREEN PIXEL for
// cascades 1, 2 and 3. Sub-pixel by construction, not by luck.
//
// Cascade 0 is deliberately excluded and keeps its exact fit. It covers the ground under the player, where
// a shadow's contact point is read most closely -- and it is also where the margin buys least, since its
// radius is small enough that six percent of it is one frame of walking.
#define SHADOW_MAP_CASCADE_PARK_MARGIN 0.06f

// How many independent caster lists a single layer may draw. Two, and only the world layer uses the second.
//
// The world layer holds the room mesh, which is cached: it is uploaded once and then only bound and drawn,
// which is what keeps a large room from re-uploading its whole mesh in every cascade of every frame. But
// scenery that the game spawns as an ACTOR belongs in that same layer -- a gate's shadow should fall on
// everything, the way a wall's does -- and an actor can move, so it cannot go in a cache keyed on geometry
// that only notices when the geometry itself changes. The Hyrule Castle gate slides open; cached, its
// shadow would stay shut.
//
// So the layer draws two lists: the cached room, and a per-frame one for scenery actors. They need separate
// buffers rather than separate calls into one, because the "already holds this list" check is what the
// caching is made of -- alternating two lists through a single buffer would defeat it on every cascade and
// hand back exactly the cost the cache exists to avoid.
#define SHADOW_MAP_CASTER_SLOTS 2
#define SHADOW_MAP_CASTER_SLOT_MAIN 0    // the room mesh in the world layer; characters in the actor layer
#define SHADOW_MAP_CASTER_SLOT_SCENERY 1 // scenery actors, world layer only, rebuilt every frame

// Per-cascade square resolution bounds. 4096 is the largest the near cascade is ever asked for, and
// anything under 256 produces texels so large that the bias needed to hide the acne swallows the
// shadow itself.
#define SHADOW_MAP_MIN_RESOLUTION 256
#define SHADOW_MAP_MAX_RESOLUTION 4096
#define SHADOW_MAP_DEFAULT_RESOLUTION 4096

// Default split distances (world units from the camera) for the four cascades. These bound the far
// plane of each cascade's ortho projection; the near plane of cascade N is the far plane of N-1.
//
// Note that the last ACTIVE split -- split[cascadeCount - 1], not split 3 -- is also the distance out to
// which the application captures casters at all. These values are the range; the cascade count selects how
// much of it is used, so lowering the count shortens the shadowed range with it.
//
// Three bands, and split 2 is therefore the end of the range. The middle band is stretched to absorb what
// the retired fourth cascade used to cover: a
// cascade's radius is dominated by the frustum's lateral spread at its far edge, so the last band is coarse
// whatever it starts at, and the way to keep the middle distance sharp is to hand more of it to the band
// before. At 4096 the three bands come out at 0.09, 0.74 and 3.6 world units per texel.
#define SHADOW_MAP_DEFAULT_SPLIT_0 150.0f
#define SHADOW_MAP_DEFAULT_SPLIT_1 1200.0f
#define SHADOW_MAP_DEFAULT_SPLIT_2 6000.0f

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
// 0.05, which is very nearly nothing -- the player is about 60 world units tall, so this is a thousandth of
// his height. It reads as a token margin rather than a working part, and that is the point: what removes
// acne now is the receiver plane bias in the shader, which compares each tap against the depth the
// receiver's OWN plane would have at that tap instead of at the kernel's centre. That correction is exact
// for a flat receiver and needs no margin, so it does not detach anything -- which is what left this term
// with nothing to do.
//
// It was 4.0, from before the plane bias existed, when this and the slope term were carrying the whole load
// between them (the normal offset is off in the tuned configuration). Every one of those units was also a
// unit of shadow sliding away from whatever cast it along the light ray, so cutting it back is bought
// contact, not lost protection. What remains covers the cases the plane correction cannot reach on its own:
// a silhouette, where the derivative straddles two surfaces and the correction is clamped.
#define SHADOW_MAP_DEFAULT_DEPTH_BIAS_WORLD 0.05f
// Slope-scaled bias, handed to the rasterizer: a multiple of the polygon's own depth gradient across a
// texel. Being relative to the gradient is exactly right -- it is nearly nothing on a surface facing the
// light and large on one edge-on to it, which is where depth runs away across a texel and acne appears.
// Being relative to the TEXEL is the part that misbehaved, since a texel of the far cascade is several
// world units, and that is now handled per cascade by the ceiling below.
//
// 1.0: one polygon gradient across one texel, which is the least this term can be and still mean anything.
// It has been 4.0 and, before that, 2.0 -- both from the era when the biases here were the only defence
// against acne. The receiver plane bias in the shader took that job over, and it does it without displacing
// anything, so the two rasterizer-side terms are now a backstop rather than the mechanism.
//
// Kept above zero rather than switched off, because the two do not overlap completely: the plane correction
// is recovered from screen-space derivatives, and at a silhouette -- where the pixel quad straddles two
// surfaces -- it is meaningless and gets clamped. This is what covers that, and being relative to the
// polygon's own gradient it costs almost nothing on the surfaces facing the light, where a shadow's contact
// point is read most closely.
//
// The per-cascade ceiling below still applies and now binds even less than before: at a quarter of the old
// multiplier, a texel would have to be four times coarser to reach it.
#define SHADOW_MAP_DEFAULT_SLOPE_BIAS 1.0f

// Front-face culling was implemented here and removed. It ends self-shadowing acne at its source rather
// than biasing it out of sight -- store only the BACK of each caster and the surface the light strikes is
// never in the map to be compared against itself -- but it requires the caster to HAVE a back. A wall
// modelled from one side, a terrain plane, a leaf quad: each has nothing left once its front is culled, and
// casts nothing at all. Too much of this game is built that way for the trade to be worth it, so the depth
// pass records both facings.

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
// Zero in the tuned configuration, which is a choice and not a fallback. Sideways is the cheapest direction
// to move a sample -- it leaves the light ray and so buys acne removal without detachment -- but it moves the
// sample off the surface in world space, and where two surfaces meet that pulls the shadow out of the corner.
// The constant bias absorbs the work instead: it detaches, but it detaches evenly, and an even offset reads
// as a soft contact where a missing corner reads as a mistake. Raise this if acne returns on curved
// geometry; the corners are what it costs.
#define SHADOW_MAP_DEFAULT_NORMAL_OFFSET 0.0f

// Ceiling on that push, in WORLD units, applied per cascade before the shader sees it.
//
// Measuring the offset in texels is right while a texel is small, but a texel of the far cascade is several
// world units, so two of them is nearly twelve -- and the shader moves the receiver's sample that far
// towards the light before comparing. The shadow lifts off whatever cast it, by more the further away the
// receiver is, because the cascade it lands in is coarser. That is peter panning that grows with distance.
//
// The ceiling only matters when the offset is on. Off, as the tuned configuration has it, nothing here
// binds: the shader multiplies by an offset of zero and the sample stays exactly on the surface.
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
// 0.4, which is where two pieces of player feedback bracket it rather than where a calculation puts it:
// 0.25 read as stair-stepped, 0.5 read as smeared, so this sits nearer the end that was complained about
// twice. Sweep it live -- it is a CVar and needs no rebuild.
//
// 0.25 failing was itself a correction. I had argued that bilinear taps cannot stair-step at any spacing;
// that was wrong. A bilinear tap removes the hardness WITHIN a texel and does nothing about the staircase
// BETWEEN texels, because the shadow edge is quantised to the texel grid either way and a one-texel ramp
// only rounds each step's corner. Hiding a staircase needs a kernel spanning several texels.
//
// So sharper distant shadows are not available from the filter. That is a texel problem, and both cures are
// priced: halve Graphics.ShadowMap.Split2 to halve the far cascade's texel at the cost of range, or double
// Graphics.ShadowMap.Resolution to halve every texel at the cost of four times the memory.
#define SHADOW_MAP_DEFAULT_FILTER_WIDTH 0.6f

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
// Not the full caster reach, because this draws geometry nothing can see and is paid in draw calls across
// the whole sphere. What sets it instead is the longest shadow the scene can throw: a caster matters behind
// the camera exactly as far back as its own shadow reaches forward. The elevation floor clamps the sun at
// thirty degrees (see MIN_ELEVATION), so that length is the caster's height over the tangent of thirty --
// about one and three quarter times its height.
//
// 800 therefore covered casters up to about four hundred and sixty units tall, which is a hut. It is not a
// tower: the Kakariko windmill stands well above that, so it left the draw list while the ground carrying
// its shadow was still in full view, and the shadow came and went with the camera angle. 1500 covers a
// caster around eight hundred and seventy units high, which is every building in the game.
//
// Raising it does not cost more capture work -- the world layer's caster list is cached and rebuilt only
// when the drawn set changes, and a wider sphere makes that set change LESS often, not more. What it costs
// is the draw itself for off-screen geometry.
#define SHADOW_MAP_DEFAULT_CASTER_DRAW_RADIUS 1500.0f

// Tallest caster whose stretched shadow is worth following, in world units. This is the OTHER half of the
// reach: the radius above is a sphere around the camera, and a sphere is the wrong shape for the question.
//
// What decides whether a caster matters is not how far away it is but whether its SHADOW arrives, and a low
// sun sends a shadow far further than the caster stands. At the elevation floor the shadow runs about twice
// the caster's height along the light -- so a tower can sit well outside any sphere anyone would pay for and
// still lay its shadow across the ground at the player's feet. Widening the sphere until it reached would
// mean drawing everything in every direction to catch the few things lying along one of them.
//
// So the test follows the light instead: keep a caster when any point of the ray it casts along passes near
// the camera, out to this much height's worth of shadow. That is a thin cylinder rather than a bigger
// sphere, and it admits the far tower whose shadow arrives while continuing to reject the equally far one
// whose shadow goes the other way.
//
// Used as a length along the light: shadow reach = this / sin(elevation), so it grows exactly when shadows
// stretch and shrinks at midday when they do not. 1200 is taller than anything in the game that stands on
// ground the player also stands on, which at the elevation floor follows a shadow about 2400 units long.
#define SHADOW_MAP_CASTER_SHADOW_HEIGHT 1200.0f

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

// How hard the shadow edge is, 0 to 1.
//
// The filter above produces a coverage value: how much of the kernel is occluded, 0 to 1. Shading with it
// directly gives a linear ramp across the whole kernel, which IS the blur. Remapping it through a narrow
// ramp centred on half coverage instead collapses that gradient into an edge, which is what a cel-shaded
// look wants anyway.
//
// This is genuinely better than simply narrowing the kernel, and for a reason worth stating: the coverage
// value varies smoothly BETWEEN texels, because every tap is bilinear. So the half-coverage contour is a
// piecewise-linear curve through the texel grid rather than a staircase along it, and hardening it keeps
// that sub-texel placement. Narrowing the kernel throws the placement away and lands back on the grid.
//
// What it costs: the wide kernel is also what antialiases the edge on screen, and an edge is by definition
// not antialiased. The remap keeps a narrow ramp rather than a step so a pixel or two of softness survives,
// but at distance -- where a texel is several world units -- a hard edge will show the contour's facets.
// That is the same texel limit as ever, traded from blurry to faceted rather than removed.
//
// 0.75: clearly an edge, with the ramp still wide enough to carry some antialiasing. 0 leaves the filtered
// gradient exactly as it was; 1 is very nearly binary.
#define SHADOW_MAP_DEFAULT_EDGE_HARDNESS 0.6f

// Edge hardness in the FURTHEST cascade, ramped to from the value above across the cascade ladder.
//
// One hardness for every distance is wrong, because the thing it is fighting is not one size. The kernel is
// three texels wide whatever the cascade, but a texel of the near cascade is 0.15 world units and one of the
// far cascade is 5.8 -- so the same hardness leaves a transition 0.14 units wide up close and 5.2 units wide
// at distance. That is the blur that would not go away.
//
// It is also why distant shadows look washed out rather than merely soft. Coverage is what the filter
// returns, and a shadow feature narrower than the kernel never reaches full coverage anywhere -- at the far
// cascade the kernel spans 17 world units, so anything thinner than that is averaged into grey and can
// never be fully dark. Thresholding at half coverage snaps it back to solid. So the two complaints are one
// artefact and one fix.
//
// 1.0, a near-binary edge in the last cascade: 5.2 world units of transition down to 1.05. Set it equal to
// SHADOW_MAP_DEFAULT_EDGE_HARDNESS for the old uniform behaviour.
#define SHADOW_MAP_DEFAULT_EDGE_HARDNESS_FAR 0.6f

// How squarely the light must strike a surface for the shadow map to be trusted on it, as the cosine of
// the angle between the surface normal and the light. Below this the shadow term is faded out entirely.
//
// This is NOT softening an artefact away -- it is declining to use data that carries no information. The
// failure it addresses is projective aliasing, and it is a different animal from acne. Acne is a false
// comparison, which the receiver plane bias fixes exactly. Projective aliasing is a sampling limit: on a
// surface turning edge-on to the light the shadow map has almost no resolution along the direction the
// surface recedes, so the shadow BOUNDARY quantises into steps of texel / sin(angle). At sixty degrees off
// that is under two world units on the middle cascade; at five degrees it is seventeen; at two, forty-two,
// which is a character's whole height. Those are the teeth. No bias moves them, because nothing is
// mis-compared -- the boundary is simply being drawn at a resolution that does not exist.
//
// Fading it out there is the physically right answer rather than a retreat: a surface edge-on to the light
// receives almost no light, so it can carry almost no shadow. Illumination and shadow-map resolution both
// go to zero together, and the term stops mattering exactly where it stops being computable.
//
// 0.35 is about seventy degrees off the normal, past which the step is already several world units.
#define SHADOW_MAP_MIN_INCIDENCE 0.35f

// Incidence at which the edge may be hardened all the way, as the same cosine. Between this and
// SHADOW_MAP_MIN_INCIDENCE the hardening tapers off, so the shadow keeps a proportionate amount of the
// filter's own gradient instead of being cut to an edge.
//
// The threshold is only as trustworthy as the contour it traces, and that contour degrades with incidence
// long before it stops carrying information altogether: the boundary quantises into steps of one texel over
// the sine of the angle, so it is already a couple of world units out at sixty degrees. Drawing a hard line
// through that prints the steps as facets. Keeping part of the blur exactly there -- and only there -- is
// what stops them, and it costs nothing on the surfaces where the boundary is well sampled, which is where
// the hard edge was wanted in the first place.
//
// 0.85 is about thirty-two degrees off the normal. Above it the step is under two world units, which the
// hardening can trace without showing a facet.
#define SHADOW_MAP_FULL_INCIDENCE 0.85f

// Floor under the incidence taper, as a fraction of the configured hardness.
//
// The taper exists because a hard contour on an undersampled boundary prints its steps as facets. But
// tapering to nothing was too far: the threshold does two things at once, and only one of them is the
// problem. It clips the faint tail of the penumbra, where the filter reports a sliver of occlusion and the
// shadow reads as a smear rather than a shadow, and it saturates the core, where a feature narrower than
// the kernel can never reach full coverage and comes out grey. Neither of those needs a hard edge -- both
// are contrast, and a ramp that keeps most of its width still delivers them.
//
// Only the last stretch towards a step is what draws facets. So the taper now bottoms out here instead of
// at zero: at 0.4 of the configured hardness the ramp still spans roughly a seventh of the coverage range
// either side of half, which clips the tail and fills in the core while staying far too wide to trace a
// texel step.
#define SHADOW_MAP_MIN_EDGE_HARDNESS_SCALE 0.4f

// Strength of the shadow where it is fully occluded (0 = invisible, 1 = black).
#define SHADOW_MAP_DEFAULT_STRENGTH 0.5f

#endif // FAST_SHADOW_MAP_H
