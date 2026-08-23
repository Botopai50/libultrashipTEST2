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
// A slice is redrawn only when its matrix changes, and the matrix changes when the cascade moves. Holding it
// still is exact -- a cascade that still contains the fitted sphere shows every caster and every receiver
// the fitted one would -- but it can only be held while there is room to hold it, and the room is exactly
// this margin. Without it the only slack was whatever the radius quantisation happened to leave, which
// measured out at 0 to 28 world units against a camera that moves ten per frame: the cascades re-parked
// every frame and nothing was ever reused.
//
// The margin is spent against ROTATION more than against walking, which is what sets these numbers. A
// cascade is centred at the middle of its band, AHEAD of the camera, so turning swings it: the far band's
// centre sits 3600 units out, and one degree of turn moves it 63. At six percent that band tolerated seven
// degrees before re-parking -- less than a flick of the stick -- which is why the pass was still redrawing
// four of five slices while the camera moved. Measured, not guessed: 3.9 of 5 against a floor of 3.
//
// The cost is that the radius sets the texel size, so this widens the penumbra by the same fraction. Stated
// as a fraction on purpose, because the absolute number is not the meaningful one: at 4096 over the default
// three-band ladder, the penumbra is already 18 screen pixels wide at the near edge of the middle band and
// 11 at the near edge of the far one, so six percent of it is about a pixel and fifteen percent is under
// three. A soft edge getting six or fifteen percent softer is not a thing an eye picks out; the same
// fraction quoted in pixels sounds much worse than it looks.
//
// (An earlier revision of this comment quoted those as "0.55, 0.55 and 0.26 of a screen pixel, sub-pixel by
// construction". That arithmetic was done on the FOUR-cascade ladder and was carried over unchecked when the
// ladder dropped to three. Retiring a cascade makes every remaining band's radius much larger relative to
// where it starts, and the true figures are the ones above -- roughly four times larger.)
//
// The last cascade gets more of it than the others. It is the one rotation hurts most in absolute terms, its
// texel is already the coarsest by an order of magnitude, and it is the furthest away -- so it has both the
// most to gain and the least to lose. Cascade 0 gets none at all and keeps its exact fit: it covers the
// ground under the player, where a shadow's contact point is read most closely, and six percent of its
// radius is one frame of walking anyway.
#define SHADOW_MAP_CASCADE_PARK_MARGIN 0.06f
#define SHADOW_MAP_CASCADE_PARK_MARGIN_FAR 0.15f

// Which margin a given cascade of an active ladder gets. First one exact, last one generous, rest in
// between. A one-cascade ladder is all "first", so it stays exact.
#define SHADOW_MAP_PARK_MARGIN_FOR(cascade, count) \
    ((cascade) == 0                                \
         ? 0.0f                                    \
         : ((cascade) + 1 >= (count) ? SHADOW_MAP_CASCADE_PARK_MARGIN_FAR : SHADOW_MAP_CASCADE_PARK_MARGIN))

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

// The actor layer's own resolution, chosen separately from the world layer's.
//
// The two layers are not the same shape of problem. The world layer covers whatever the cascade covers and
// is reused across frames when the camera holds still; the actor layer holds a few thousand triangles of
// animating characters standing in one small part of that area, and its content key changes every frame, so
// it is cleared and redrawn every frame no matter what. Measured in a modded save it was 2 of the 4.6 slices
// redrawn per frame -- and a slice is a fixed cost in fill, whatever is drawn into it, because the clear
// alone writes the whole surface (33.6 MB at 4096, D16).
//
// Spending the world layer's resolution on that is the waste this separates out: characters are small, near,
// and shadowed onto ground the player is standing on, so their map is oversampled at 4096 by a wide margin.
//
// Defaults to matching the world layer, so the split changes nothing until it is asked for. When the two are
// equal the backend binds ONE array to both slots and the result is identical to having no split at all --
// the second array only exists once the resolutions differ.
#define SHADOW_MAP_DEFAULT_ACTOR_RESOLUTION SHADOW_MAP_DEFAULT_RESOLUTION

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
#define SHADOW_MAP_DEFAULT_SPLIT_0 350.0f
#define SHADOW_MAP_DEFAULT_SPLIT_1 2500.0f
#define SHADOW_MAP_DEFAULT_SPLIT_2 6000.0f

// Fraction of a cascade's range over which it cross-fades into the next one. The shader samples both
// maps across this band and blends with smoothstep, which is what keeps the resolution change from
// showing up as a hard line sweeping across the ground as the camera moves ("cascade popping").
// Expressed as a fraction so the band scales with each cascade's size.
#define SHADOW_MAP_DEFAULT_BLEND_FRACTION 0.1f

// Slope-scaled bias, handed to the rasterizer: a multiple of the polygon's own depth gradient across a
// texel. Being relative to the gradient is exactly right -- it is nearly nothing on a surface facing the
// light and large on one edge-on to it, which is where depth runs away across a texel and acne appears.
// Being relative to the TEXEL is the part that misbehaved, since a texel of the far cascade is several
// world units, and that is now handled per cascade by the ceiling below.
//
// This is the ONLY offset in the system, and it is fixed rather than settable. Everything downstream
// compares a receiver's depth against the stored depth directly -- no constant bias, no normal offset, no
// receiver-plane correction -- so without this every surface shadows itself and the scene stripes. That is
// geometry, not a preference, which is why it is not a knob.
//
// It is also the cheapest place for the offset to live. Being relative to the polygon's gradient it is
// nearly nothing on a surface facing the light, which is where a shadow's contact point is read most
// closely, and large only on one edge-on to it, where depth runs across a texel in a step and acne would
// otherwise appear.
//
// 1.0 is one polygon gradient across one texel, the least this term can be and still mean anything. It has
// been 4.0 and 2.0 in earlier revisions, when a stack of other bias terms sat on top of it; with those gone
// this may well need to move. It is a starting point, not a tuned value.
#define SHADOW_MAP_SLOPE_BIAS 1.0f

// Front-face culling was implemented here and removed. It ends self-shadowing acne at its source rather
// than biasing it out of sight -- store only the BACK of each caster and the surface the light strikes is
// never in the map to be compared against itself -- but it requires the caster to HAVE a back. A wall
// modelled from one side, a terrain plane, a leaf quad: each has nothing left once its front is culled, and
// casts nothing at all. Too much of this game is built that way for the trade to be worth it, so the depth
// pass records both facings.

// Ceiling on what that slope term may displace a receiver by, in WORLD units.
//
// The rasterizer takes one slope value per state, not per draw, so this cannot be capped per pixel -- the
// backend builds a separate rasterizer state per cascade instead, each with the slope reduced to whatever
// keeps its own texel under this ceiling.
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

// Floor on how low the key light may sit before the cascades are built from it, as the sine of its angle
// above the horizon (0 = the horizon itself, 1 = straight overhead). A light near the horizon stretches
// every shadow towards infinity, which reads as wrong long before it is geometrically wrong -- and it also
// wastes the cascade, since the projection has to cover a footprint far longer than the scene it is
// shading. The light's compass bearing is preserved; only its height is lifted.
// 0.60 is about 37 degrees, which caps a shadow at roughly 1.3x the caster's height. It was 0.5 (30
// degrees, 1.7x); raising it shortens the longest shadows the scene can throw, which is worth more here than
// it looks -- a shadow's boundary quantises into steps of one texel over the sine of the angle between the
// receiver and the light, so a sun held higher is also a sun whose shadows are sampled better.
#define SHADOW_MAP_DEFAULT_MIN_ELEVATION 0.60f

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
// of the angle. 0.999999 is about a twelfth of a degree.
//
// The cascade centre is snapped to whole texels along the LIGHT's own axes, which is what stops shadow edges
// shimmering as the camera moves. That only works if those axes hold still: the game's environment light
// turns continuously with the time of day, and a basis that turns with it is a grid that turns with it, so
// the snapping would be measuring against a ruler that keeps rotating.
//
// So the direction is held, and released in one step when it has drifted past this threshold. The cost is
// that the whole grid re-aligns at once when that happens -- and this number decides whether that step is a
// detail or the loudest thing on screen.
//
// It was 0.9994, about two degrees, and two degrees is enormous. What moves is the shadow's TIP, by roughly
// the caster's height over sin squared of the sun's elevation, times the angle: at the default elevation
// floor that is 5.7 world units for Link, 19 for a wall, and 57 for a castle tower -- Link's whole height,
// arriving in a single frame, once every eighty frames. The jump scales with the caster, which is exactly
// why it reads as "everything except the actors": a character's shadow is short and its own movement covers
// the step, while the architecture's shadow is long and visibly jolts.
//
// A twelfth of a degree puts those at 0.23, 0.78 and 2.34 units, under one texel of the cascade the
// architecture is usually in, and it fires every three frames instead of every eighty. That reads as motion
// rather than as a jolt.
//
// Firing more often costs cascade rebuilds, and measurement says that cost is already spent: profiling this
// scene showed all five slices redrawn on 60 of 60 frames during play, so the cascades were never being held
// still by this anyway. It was buying grid stability alone, and buying far more than the shimmer was worth.
#define SHADOW_MAP_LIGHT_DIR_HYSTERESIS_COS 0.999999f

// Strength of the shadow where it is fully occluded (0 = invisible, 1 = black).
#define SHADOW_MAP_DEFAULT_STRENGTH 0.5f

// How often each cascade is rebuilt, as a divisor of the frame rate: 1 rebuilds every frame, 2 every other
// frame, and so on. At 60 fps the defaults below run the near and mid cascades at 60 Hz and the far one at
// 30 Hz.
//
// The far cascade is the one worth halving. It covers the largest area for the same number of texels, so
// its content changes the least per frame -- a tree three thousand units away moves a fraction of a texel
// between frames, while the near cascade's shadows track the player directly and any hitch there is seen.
// It is also the most expensive slice to fill, because its footprint sweeps in the most casters.
//
// Skipping a rebuild means freezing that cascade's MATRIX too, not just declining to draw. The slice still
// holds depths rendered through the matrix it was last drawn with, and reading them through a matrix that
// has since moved with the camera projects the shadow from where the light used to be. That is not a stale
// shadow but a misplaced one -- a worse artefact than the cost being saved. The interpreter therefore
// writes back the held matrix on skipped frames, which has the useful side effect of making the existing
// content-key reuse test skip the draw on its own: a cascade whose matrix and caster list are unchanged is
// already known to hold what a redraw would produce.
//
// The cost of a divisor above 1 is shadow lag under motion, worst when the camera turns quickly. Keep the
// near cascade at 1; it is the one the eye is on.
#define SHADOW_MAP_DEFAULT_CASCADE_DIVISOR_0 1
#define SHADOW_MAP_DEFAULT_CASCADE_DIVISOR_1 1
#define SHADOW_MAP_DEFAULT_CASCADE_DIVISOR_2 2

// Highest divisor the menu offers. Beyond about four the lag is visible on anything that moves, and the
// saving has already flattened out -- a cascade drawn every fourth frame costs a quarter of its full rate,
// and every further step buys a smaller fraction of a smaller number.
#define SHADOW_MAP_MAX_CASCADE_DIVISOR 4

// How many frames the world-caster capture stays armed after the last change to its signature.
//
// The capture is armed when a change is detected and recorded on the following frame -- the signature is
// only complete once the frame has drawn, by which point it is too late to record that frame. Armed for
// exactly one frame, a caster that moves continuously is therefore captured on every OTHER frame: it
// changes, arms, is captured, changes again, arms again. Half rate, from a mechanism whose whole purpose is
// to decide when NOT to redraw.
//
// Holding the arm for a short run of frames after the last change fixes it: while something is moving the
// signature keeps changing, the window keeps being refilled, and the capture lands on every frame. When the
// movement stops the window drains and the cache costs nothing again.
//
// Six frames is a tenth of a second at sixty -- long enough to bridge an object that pauses mid-animation
// without holding the capture open behind a scene that has genuinely settled.
#define SHADOW_MAP_WORLD_SETTLE_FRAMES 6

// Highest debug view the receiver shader recognises. The application passes a view number through
// GfxRenderingAPI::SetShadowMapParams; anything outside 0..this shades normally.
//
// Worth saying plainly, because it is the lesson a long line of removed constants was tuned without: a
// setting cannot localise an artefact. Every one of them acted on the output of the depth comparison, and
// at that point a term that is wrong is indistinguishable from a term that is right but badly scaled --
// both are just a number that came out too dark. So a shape-wrong artefact (faceted, triangular, stepped
// edges, as opposed to a shadow in the wrong place) sends you tuning, and tuning can only trade it against
// a different artefact. That loop is why the receiver is now a depth comparison and a threshold and
// nothing else: a baseline that can be reasoned about beats a stack of mitigations that cannot.
//
// The views print the shader's INPUTS unshaded, so the question stops being "does this setting help" and
// becomes "is this value the shape of the thing on screen":
//   1  everything outside a cascade's footprint painted occluded    (output)
//   2  the two caster layers separated by colour                    (output)
//   3  the receiver normal                                          (input)
//   4  where that normal came from, vertex or recovered face        (input)
//   5  the filter's raw coverage, before the hardening remap        (input)
//   7  the cascade each pixel sampled                               (input)
// 6, 8 and 9 were retired with the machinery they measured; the numbering of the rest is deliberately
// unchanged, so 5 still means what it meant. The shader's PSMain carries the reading order -- which view to
// check first, and what each answer rules out. Keep this bound in step with the arms implemented there.
#define SHADOW_MAP_MAX_DEBUG_VIEW 7

#endif // FAST_SHADOW_MAP_H
