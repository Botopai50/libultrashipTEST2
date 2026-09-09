#ifndef FAST_SHADOW_MAP_H
#define FAST_SHADOW_MAP_H

#include <math.h> // powf, for the split ladder below

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

// Levels a clipmap may have. Declared here rather than beside the rest of its contract because
// the slice arithmetic below needs it; the reasoning is with SHADOW_MAP_DEFAULT_CLIPMAP_LEVELS.
#define SHADOW_MAP_MAX_CLIPMAP_LEVELS 10

// Levels either layout may ask for, and therefore the size of everything held per level: the fitted
// matrices, the parking state, the update divisors. The clipmap is the one that wants more.
#define SHADOW_MAP_MAX_LEVELS \
    (SHADOW_MAP_MAX_CASCADES > SHADOW_MAP_MAX_CLIPMAP_LEVELS ? SHADOW_MAP_MAX_CASCADES \
                                                             : SHADOW_MAP_MAX_CLIPMAP_LEVELS)

// Slices are laid out world-layer-first: world cascade C is slice C, actor cascade C is slice
// cascadeCount + C. The actor half is the shorter one, so the total is not a simple product.
#define SHADOW_MAP_ACTOR_CASCADES_FOR(count) ((count) < SHADOW_MAP_ACTOR_CASCADES ? (count) : SHADOW_MAP_ACTOR_CASCADES)
#define SHADOW_MAP_SLICES_FOR(count) ((count) + SHADOW_MAP_ACTOR_CASCADES_FOR(count))

// The TEXTURE array has to hold whichever layout asks for more slices, and the clipmap asks for more: six
// levels against three cascades (see SHADOW_MAP_MAX_CLIPMAP_LEVELS, further down).
//
// Only the texture grows. The constant buffer's matrix array stays at SHADOW_MAP_MAX_CASCADES, because the
// clipmap has no per-level matrix to put in it -- every level shares one basis and differs by a power of
// two, so its projection is arithmetic in the shader. That is the difference between this being a change to
// an allocation and a change to every shader that samples a shadow.
#define SHADOW_MAP_MAX_SLICES                                                    \
    (SHADOW_MAP_SLICES_FOR(SHADOW_MAP_MAX_CASCADES) >                            \
             SHADOW_MAP_SLICES_FOR(SHADOW_MAP_MAX_CLIPMAP_LEVELS)                \
         ? SHADOW_MAP_SLICES_FOR(SHADOW_MAP_MAX_CASCADES)                        \
         : SHADOW_MAP_SLICES_FOR(SHADOW_MAP_MAX_CLIPMAP_LEVELS))

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

// Per-cascade square resolution bounds. Anything under 256 produces texels so large that the bias needed
// to hide the acne swallows the shadow itself.
//
// 4096 is the ceiling, and it is a measured one rather than a hardware limit -- feature level 10.0
// guarantees 8192, and 8192 was offered briefly and taken back out for being unusable in practice. A D16
// slice quadruples with each step: 8 MB at 2048, 32 at 4096, 128 at 8192. Three cascades at 8192 is 640 MB
// with the actor layer, and eight clipmap levels at 8192 is over a gigabyte.
//
// The clipmap's answer to wanting sharper shadows is MORE LEVELS at a moderate resolution, not fewer at an
// extreme one: levels cost linearly and a level does not have to be large to be sharp. That is the knob
// this ceiling is pointing at.
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
#define SHADOW_MAP_DEFAULT_BLEND_FRACTION 0.2f

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


// ===================================================================================================
// SOH [Enhancement] Edge quality: the five techniques that attack a stair-stepped shadow edge.
//
// Everything above this line describes WHERE a shadow is. Everything below describes what its EDGE looks
// like once it is in the right place, which is a separate question and was the one left unanswered: the
// receiver samples a single bilinear quad, so the penumbra is exactly one texel wide and the boundary can
// only move in whole-texel steps. At the default ladder that step is 0.09, 0.74 and 3.6 world units in the
// three bands -- one to two screen pixels through most of the useful range, which is read as a staircase.
//
// Four independent techniques, each switchable on its own and each with its own tuning, because they attack
// the same artefact at different points in the pipeline and stack rather than compete:
//
//   Analytic edge  -- receiver.  Recovers the sub-texel position of the boundary inside the quad already
//                                fetched. No extra taps.
//   Jitter         -- receiver.  Rotates tap offsets per pixel, trading the step for dither.
//   Filterable map -- map.       Stores a filterable quantity (ESM/VSM/MSM) and blurs the map itself, so
//                                the receiver stays one fetch and softness stops costing taps.
//   Ladder         -- fit.       Redistributes cascade range so the texel size is uniform instead of
//                                350/2500/6000's 40x spread.
//
// The policy split is the same as everything above: the framework owns the technique, the application owns
// the tuning and pushes it in as one struct. No app-specific CVar keys live here, only the defaults and the
// bounds the framework will honour.
// ===================================================================================================

// --- Technique 2: analytic edge reconstruction -----------------------------------------------------
//
// The receiver already fetches the 2x2 quad of stored depths around the sample point and compares each.
// Bilinear-weighting those four binary results gives a boundary that is continuous but only one texel
// wide, and whose iso-contour follows the texel grid's own diagonals -- which is the staircase.
//
// Reconstructing instead: the four comparisons say which corners of the quad are occluded, and the four
// depths say by how much each corner misses. Together they locate where inside the quad the occluding
// surface crosses the receiver, and coverage can be computed from that crossing directly. No extra fetch,
// and the contour stops being a function of the grid.
//
// Exact for a single straight boundary through the quad, which covers walls, steps, roofs and platform
// edges -- the geometry the staircase is complained about on. It degrades to the bilinear answer where the
// quad holds more than one boundary (foliage), which is the correct place to give up.
#define SHADOW_MAP_DEFAULT_ANALYTIC_EDGE 1

// How far the reconstructed coverage ramp is spread, in texels. 1.0 is the geometric answer -- the ramp
// occupies exactly the texel the boundary crosses. Above that it is deliberately widened, which is the
// cheapest softening available anywhere in the system since it costs arithmetic and no fetches.
#define SHADOW_MAP_DEFAULT_ANALYTIC_EDGE_WIDTH 2.0f
#define SHADOW_MAP_MAX_ANALYTIC_EDGE_WIDTH 4.0f

// --- Technique 3: stochastic jitter ----------------------------------------------------------------
//
// Rotate the tap pattern by a per-pixel angle instead of holding it on the grid. The step does not get
// smaller, but it stops being the SAME step for neighbouring pixels, so the eye reads dither rather than a
// staircase. Costs one hash and a rotation; the tap count is what it is.
//
// The honest caveat, stated here because it decides whether this is worth switching on: this renderer has
// FXAA and no temporal accumulation. Dither with nothing to average it over is grain, and grain in motion
// is its own artefact. It is offered because on a still image at a moderate tap count it is clearly better
// than the staircase, and because whether the trade is acceptable is a matter of taste that a constant
// cannot settle.
#define SHADOW_MAP_DEFAULT_JITTER 1

// Taps in the rotated pattern. Each is a full bilinear quad fetch, so this is the one knob here that costs
// real bandwidth, and it is also what decides whether the dither reads as softness or as noise.
#define SHADOW_MAP_DEFAULT_JITTER_TAPS 8
#define SHADOW_MAP_MAX_JITTER_TAPS 16

// Radius of the rotated pattern, in texels of the cascade being sampled.
#define SHADOW_MAP_DEFAULT_JITTER_RADIUS 2.0f
#define SHADOW_MAP_MAX_JITTER_RADIUS 8.0f

// --- Static caster cache ------------------------------------------------------------------------
//
// A slice's casters divide into two kinds that behave nothing alike. The room mesh is a cache: uploaded
// once and rebuilt only when the scene itself changes. The scenery ACTORS are per frame -- a gate slides,
// a tree sways -- and cannot be cached on geometry, because the geometry is the same and the matrix is not.
//
// Today a slice holds both and is redrawn as a unit. So one swaying tree costs the whole room mesh again,
// in every slice the tree reaches, every frame it moves. That is the case this exists for. It is not a
// corner case: chunking the scenery list was already added to stop a single tree invalidating all three
// world cascades, which says how often it happens.
//
// THE SPLIT. Keep a second copy of each world slice holding the STATIC casters alone. Each frame, copy that
// into the live slice and draw only what moved on top. The copy replaces both the clear and the room mesh's
// rasterisation.
//
//   4096  a slice costs about 420 us to redraw; the copy is about 62 us
//   1024  the copy is about 4 us, which is nothing
//
// WHAT IT COSTS is a second array the size of the world layer's:
//
//   cascades at 4096   +96 MB      cascades at 2048   +24 MB
//   clipmap at 1024    +12 MB      clipmap at 2048    +48 MB
//
// Which is why it is off by default and why it belongs beside the clipmap: the layout that wants many
// small levels is exactly the one where a second copy is cheap. At 4096 cascades this is a real 96 MB, and
// worth it only in a scene with scenery that actually moves.
//
// WHAT IT DOES NOT HELP. A settled scene already redraws nothing -- the parking and the content keys see to
// that -- so this changes nothing there. It buys back the cost of movement, not the cost of standing still.
#define SHADOW_MAP_DEFAULT_STATIC_CACHE 0

// What ShadowMapBeginCascadeSplit answers, and therefore what the caller must draw.
#define SHADOW_MAP_SLICE_REUSED 0  // nothing at all: the live slice already holds this image
#define SHADOW_MAP_SLICE_FULL 1    // draw the static casters, then the dynamic ones
#define SHADOW_MAP_SLICE_DYNAMIC 2 // the static half was copied in; draw only what moved

// --- Shadow clipmap ------------------------------------------------------------------------------
//
// A second way of laying the map out, chosen instead of the cascade ladder. The idea is the directional
// half of Unreal's Virtual Shadow Maps, minus the part that does not port.
//
// WHAT A CLIPMAP IS. Rather than N slabs fitted to N slices of the view frustum, it is N nested squares
// CENTRED ON THE CAMERA, each exactly twice the world extent of the one inside it. Level 0 is small and
// fine; each level out doubles its extent and therefore doubles its texel. Which level a pixel reads is a
// function of how far it is from the centre, not of which band of view depth it fell in.
//
// WHY IT IS BETTER HERE, and it is not a small difference:
//
//   The texel ratio between neighbours is exactly 2. The hand-fitted ladder's is 3 to 8 -- measured at
//   0.09, 0.74 and 3.6 world units -- and that jump is what a cross-fade has to hide. A factor of two is
//   most of the way to invisible before any blending is applied.
//
//   The boundaries are circles around the camera, so they travel WITH the player instead of sweeping
//   across the world as the camera moves. A seam that moves with you is one you stop noticing.
//
//   Levels can be far smaller. Density is uniform, so a level does not need 4096 to be sharp where it
//   matters: six levels at 1024 is 6.3 megatexels against three at 4096's 50. Eight times less memory and
//   eight times less fill, better distributed.
//
// WHAT IS NOT PORTED, and why. Unreal's version is virtual in the memory sense: a page table, a physical
// page pool, and per-page allocation driven by which pages the visible pixels actually touch. That needs
// compute shaders, indirect draw, atomics and a depth prepass, and it is practical there because Nanite
// can re-rasterise geometry cheaply. This renderer has none of those -- the receiver is compiled by FXC
// inside the frame, and the geometry is N64 display lists resubmitted per level. The LAYOUT is the part
// that ports, and the layout is where the image quality lives.
//
// THE SHADER HAS NO ARRAY, which is what makes this affordable at ps_4_0 rather than merely desirable.
// Every level shares one light basis and differs only by a power of two, so the level, its extent, its
// texel and its snapped centre are all ARITHMETIC -- there is no per-level matrix to index, no comparison
// chain over splits, and no dynamically indexed constant-buffer array (which ps_4_0 does not allow, and
// which tools/shader-validate cannot warn about because Wine's compiler accepts it). The clipmap receiver
// is smaller than the cascade one despite covering twice as many levels.
#define SHADOW_MAP_LAYOUT_CASCADE 0 // the fitted ladder. What everything above describes.
#define SHADOW_MAP_LAYOUT_CLIPMAP 1
#define SHADOW_MAP_LAYOUT_MAX 1
#define SHADOW_MAP_DEFAULT_LAYOUT SHADOW_MAP_LAYOUT_CASCADE

// Levels in the clipmap, and the knob to reach for when the range is short.
//
// Levels cost LINEARLY and buy range EXPONENTIALLY, which is the whole shape of this ladder and is the
// opposite of how the cascade count behaves. One more level doubles the reach for the price of one more
// slice -- and a clipmap slice is small, because uniform density means no level has to be large to be
// sharp. Measured across the plausible settings:
//
//   levels  base   res    reach    texel L0   memory
//        6   190  4096     6080       0.093    192 MB
//        8   120  2048    15360       0.117     64 MB
//       10   100  2048    51200       0.098     80 MB
//
// So more levels at a lower per-level resolution beats fewer at a high one on every axis that matters.
// Ten is the bound because the depth pass still costs per slice and every level past the horizon is a
// slice drawn for nothing.
#define SHADOW_MAP_DEFAULT_CLIPMAP_LEVELS 8

// Half the world extent of level 0, in world units. The whole ladder follows: level i is this times 2^i,
// and the outermost reaches base * 2^(levels-1).
//
// 120 with eight levels reaches about 15000 -- two and a half times the cascade ladder -- while level 0 is
// 240 units across, which at 2048 is 0.117 world units per texel underfoot.
#define SHADOW_MAP_DEFAULT_CLIPMAP_BASE 120.0f
#define SHADOW_MAP_MIN_CLIPMAP_BASE 20.0f
#define SHADOW_MAP_MAX_CLIPMAP_BASE 2000.0f

// Per-level square resolution for the clipmap, and its OWN setting rather than the cascade ladder's.
//
// It was a dead constant at first -- declared, and the clipmap left sharing Graphics.ShadowMap.Resolution
// with the cascades. That coupling is wrong in both directions: a clipmap wants many small levels and a
// cascade ladder wants few large ones, so a resolution that suits one starves or bankrupts the other. Six
// levels at the cascade default of 4096 is 192 MB, which is most of why raising the level count looked
// unaffordable when it is in fact the cheapest knob here.
//
// 4096, and 2048 was measured to be a regression rather than a saving. Against the cascade ladder at its
// own 4096, in world units per texel at a given distance from the camera:
//
//     dist   ladder   clip@2048   clip@4096
//      150    0.090       0.234       0.117
//      400    0.740       0.469       0.234
//      900    0.740       0.938       0.469
//     2000    0.740       3.750       1.875
//     4000    3.600       7.500       3.750
//
// At 2048 the clipmap loses at nearly every distance -- five times worse at 2000 units. At 4096 it wins or
// ties almost everywhere. A layout meant to replace the ladder must not be coarser than it.
//
// The reason it needs the higher number is structural and worth stating, because no setting fixes it: a
// clipmap level is a SQUARE CENTRED ON THE CAMERA and a cascade is a slab FITTED TO THE VIEW FRUSTUM. The
// camera looks one way, so the square spends most of its area on ground nobody is looking at. That is the
// price of boundaries that travel with the player and of a texel ratio of exactly two; the density has to
// be bought back with resolution.
//
// Eight levels at 4096 is 256 MB for the world layer. The actor layer's two levels cover only the innermost
// squares -- a few hundred units -- so 4096 there is extravagant, and Resolução (Personagens) is a separate
// setting for exactly this reason: dropping it to 1024 takes 60 MB off the total.
#define SHADOW_MAP_DEFAULT_CLIPMAP_RESOLUTION 4096

// --- Edge hardening ------------------------------------------------------------------------------
//
// Everything above widens or smooths the boundary. This is the control in the other direction: find the
// boundary and compress it, for a harder, more defined outline.
//
// The "finding" is free and needs no extra fetches, because coverage already carries it. A value at 0 or 1
// is interior -- fully shadowed or fully lit -- and only the boundary produces anything in between. So the
// remap below acts ONLY on the edge by construction: interiors map to themselves whatever the setting, and
// the width of the ramp is the whole of what changes.
//
// Applied AFTER the raw coverage is taken for diagnostics, which is what debug view 5 has always promised
// -- it shows "the filter's raw coverage, before the hardening remap". There was one before the rollback
// and the view outlived it; this puts the value the view describes back under it. So a boundary that is
// faceted in view 5 and faceted on screen is faceted in the comparison, and one that is smooth in view 5
// and hard on screen is this.
#define SHADOW_MAP_DEFAULT_EDGE_HARDEN 0

// How far the ramp is compressed towards a step. 0 leaves coverage exactly as it arrived; 1 is a hard
// threshold with no transition at all.
//
// A hard threshold is not automatically what is wanted: the penumbra carries the cascade's texel size, so
// removing it removes the only cue that distance is being sampled more coarsely, and the boundary starts
// showing the texel grid it was hiding. Somewhere short of 1 is usually where this lands.
#define SHADOW_MAP_DEFAULT_EDGE_HARDNESS 0.5f

// Where in the coverage range the boundary is taken to be. 0.5 is the geometric answer -- half the kernel
// occluded is the edge.
//
// Moving it GROWS or SHRINKS the shadow: below 0.5 a lightly-occluded pixel counts as shadowed and the
// shadow spreads; above it the shadow pulls in. Worth having next to the hardness, because compressing a
// ramp around a shifted centre is how an outline is thickened or thinned rather than merely sharpened.
#define SHADOW_MAP_DEFAULT_EDGE_THRESHOLD 0.5f

// --- Technique 4: cascade split ladder -------------------------------------------------------------
//
// Where the cascade boundaries fall, which decides the texel size in each band and therefore how big the
// staircase step is before any filtering touches it.
//
// The hand-drawn ladder (350 / 2500 / 6000) spends the near cascade on a band so short that its texel is
// 0.09 world units -- finer than anything can be seen at that distance -- and then hands the middle band a
// texel eight times coarser and the far band forty times coarser. Uniforming that spread is free: it is
// three numbers, no shader and no new memory, and it shrinks the step exactly where the step is visible.
#define SHADOW_MAP_LADDER_MANUAL 0    // whatever the application's split sliders say. The existing behaviour.
#define SHADOW_MAP_LADDER_PRACTICAL 1 // blend of uniform and logarithmic, by lambda below.
#define SHADOW_MAP_LADDER_MAX 1
#define SHADOW_MAP_DEFAULT_LADDER_MODE SHADOW_MAP_LADDER_MANUAL

// Blend between a uniform ladder (0) and a logarithmic one (1), the standard "practical split scheme".
//
// Logarithmic is what makes the texel size uniform across bands, which is the point; pure logarithmic
// however puts the first split extremely close to the camera, and a cascade that covers almost nothing
// wastes a whole slice. The blend is the usual compromise and 0.75 is where it is normally landed.
#define SHADOW_MAP_DEFAULT_LADDER_LAMBDA 0.85f

// The near distance the ladder is generated from. Not the camera's actual near plane, which is small
// enough to drag the first split down to nothing; this is the distance at which shadows start being worth
// resolving finely.
#define SHADOW_MAP_DEFAULT_LADDER_NEAR 40.0f

// Generate the split ladder for `count` cascades out to `farDistance`, into splits[0..count-1].
//
// The practical split scheme: split i is a blend of the uniform ladder (near + (far-near) * i/N, which
// keeps each band the same DEPTH) and the logarithmic one (near * (far/near)^(i/N), which keeps each band
// the same RATIO and therefore each texel the same size). Lambda picks between them.
//
// Lives in the header rather than in the interpreter because the menu wants to show the numbers it is about
// to produce, and a preview that reimplements the formula is a preview that can disagree with it.
//
// A no-op when mode is MANUAL: the caller's splits are left exactly as they arrived.
static inline void ShadowMapLadderSplits(int mode, float lambda, float nearDistance, float farDistance, int count,
                                         float* splits) {
    int i;
    if (splits == 0 || count < 1 || mode != SHADOW_MAP_LADDER_PRACTICAL) {
        return;
    }
    if (nearDistance < 1.0f) {
        nearDistance = 1.0f;
    }
    if (farDistance <= nearDistance) {
        farDistance = nearDistance + 1.0f;
    }
    for (i = 0; i < count; i++) {
        const float fraction = (float)(i + 1) / (float)count;
        const float uniform = nearDistance + ((farDistance - nearDistance) * fraction);
        const float logarithmic = nearDistance * powf(farDistance / nearDistance, fraction);
        splits[i] = (lambda * logarithmic) + ((1.0f - lambda) * uniform);
    }
    // The last split IS the range (see SHADOW_MAP_DEFAULT_SPLIT_2), so it must land exactly on it rather
    // than on whatever the blend rounds to -- the caster capture reads this number.
    splits[count - 1] = farDistance;
}

// --- Shadow acne -----------------------------------------------------------------------------------
//
// A surface shadowing itself. The depth pass records a surface at one depth; the receiver asks about the
// same surface and gets an answer a fraction off, and half the texels come back "occluded". At a grazing
// view angle the striping projects into long rays converging at the horizon, which is why it reads as a
// starburst on the ground rather than as stripes.
//
// The system's standing defence is the rasterizer's slope-scaled bias (SHADOW_MAP_SLOPE_BIAS), applied
// while the depth map is written, and for the ordinary receiver it is enough: that receiver's world
// position is the interpolated vertex position, which is the exact surface the depth pass rasterised.
//
// These exist for where it is not enough -- a cascade whose texel has grown very large, a surface almost
// edge-on to the light, a scene the ladder is stretched across. They are off by default because the system
// does not normally need them.
//
// Each is a different place to intervene, and they compose:
//
//   Normal offset -- move the sample point off the surface along its own normal, before projecting. The
//                    only one that is correct in principle rather than a fudge: the error being corrected
//                    is a displacement in world space, and this is a displacement in world space. Costs
//                    nothing at the contact point, because the offset is along the surface, not along the
//                    light -- so it does not detach a shadow from its caster the way a depth bias does.
//
//   Light offset  -- move the sample point toward the light. Simple, and the classic cause of "peter
//                    panning": push far enough to clear the acne and a shadow visibly parts from the foot
//                    of the thing casting it.
//
//   Depth bias    -- subtract from the receiver's depth after projecting. Cheapest, and the least
//                    discriminating: it acts the same on a surface facing the light, where there was
//                    never any acne to remove, as on one edge-on to it.
//
//   Slope scaling -- multiply whichever of the above are on by how edge-on the surface is to the light.
//                    Acne is a grazing-angle problem, so scaling by the angle spends the correction where
//                    it is needed and nearly nothing where it is not.

// Whether the corrections run at all.
//
// ON, and these values are a player's, taken from a tuned config rather than reasoned to. The rasterizer's
// own slope bias is the standing defence and is usually enough; where it was not, the combination below --
// a small normal offset, slope-scaled, and nothing else -- was what cleared it in practice.
#define SHADOW_MAP_DEFAULT_ACNE_ENABLED 1

// Normal offset, in multiples of the sampled cascade's texel. Expressed in texels rather than world units
// because the error it corrects is itself a texel-sized quantity -- a fixed world offset would be far too
// large in the near cascade and far too small in the far one.
#define SHADOW_MAP_DEFAULT_ACNE_NORMAL_OFFSET 1
#define SHADOW_MAP_DEFAULT_ACNE_NORMAL_TEXELS 0.6f
#define SHADOW_MAP_MAX_ACNE_NORMAL_TEXELS 8.0f

// Scale the corrections by how edge-on the surface is to the light, as 1 - N.L clamped by the ceiling
// below. On by default because it is what keeps the corrections from acting where there is no acne.
//
// The ceiling matters: 1/(N.L) runs to infinity as a surface turns edge-on, and an unbounded offset there
// throws the sample point far enough to sample a different part of the scene entirely.
#define SHADOW_MAP_DEFAULT_ACNE_SLOPE_SCALED 1
#define SHADOW_MAP_DEFAULT_ACNE_SLOPE_MAX 3.5f
#define SHADOW_MAP_MAX_ACNE_SLOPE_MAX 10.0f

typedef struct ShadowMapAcne {
    int enabled;           // 0/1 -- master switch for every correction below
    int normalOffset;      // 0/1
    float normalTexels;    // multiples of the sampled cascade's texel
    int slopeScaled;       // 0/1 -- scale the offset above by how edge-on the surface is
    float slopeMax;        // ceiling on that scale
} ShadowMapAcne;

// --- The struct the application pushes -------------------------------------------------------------
//
// One struct rather than twenty arguments, because these travel together through five layers (menu ->
// per-frame snapshot -> interpreter -> rendering API -> constant buffer) and adding a knob should not mean
// editing five signatures. Plain C layout: this header is included from both sides of the C boundary.
//
// Zero-initialising this gives every technique OFF and every tuning at zero, which is not the same as the
// defaults -- call ShadowMapQualityDefaults() rather than relying on {}.
#define SHADOW_MAP_DEFAULT_SMSR 0
#define SHADOW_MAP_DEFAULT_SMSR_STEPS 16
#define SHADOW_MAP_MAX_SMSR_STEPS 64
#define SHADOW_MAP_DEFAULT_SMSR_EPSILON 0.00002f
#define SHADOW_MAP_MAX_SMSR_EPSILON 0.001f

typedef struct ShadowMapQuality {
    int smsr;             // binary silhouette revectorization, bypasses all shadow filtering
    int smsrMaxSteps;      // maximum texels traversed in each direction
    float smsrEpsilon;     // near-equality tolerance in normalized shadow depth
    // Technique 1 -- filterable maps

    // Technique 2 -- analytic edge
    int analyticEdge;         // 0/1
    float analyticEdgeWidth;  // texels

    // Technique 3 -- stochastic jitter
    int jitter;          // 0/1
    int jitterTaps;      // 1..SHADOW_MAP_MAX_JITTER_TAPS
    float jitterRadius;  // texels

    // Layout -- the cascade ladder, or the clipmap
    int layout;           // SHADOW_MAP_LAYOUT_*
    int staticCache;      // 0/1 -- keep a static-only copy of each world slice and blit it


    int clipmapLevels;      // clipmap only
    float clipmapBase;      // half-extent of level 0, world units
    int clipmapResolution;  // per-level square resolution, independent of the cascade ladder's

    // Edge hardening
    int edgeHarden;       // 0/1
    float edgeHardness;   // 0 = unchanged, 1 = a hard threshold
    float edgeThreshold;  // where the boundary sits in the coverage range

    // Technique 4 -- split ladder
    int ladderMode;      // SHADOW_MAP_LADDER_*
    float ladderLambda;  // 0 = uniform, 1 = logarithmic
    float ladderNear;    // world units


    // Shadow acne. Travels with the rest rather than in its own setter: it is pushed once per frame from
    // the same place and adding a second path through five layers would buy nothing.
    ShadowMapAcne acne;
} ShadowMapQuality;

// The defaults above, as a value. Written as a function rather than an initialiser macro so both sides of
// the C boundary get the same one and it cannot drift.
static inline ShadowMapQuality ShadowMapQualityDefaults(void) {
    ShadowMapQuality q;
    q.smsr = SHADOW_MAP_DEFAULT_SMSR;
    q.smsrMaxSteps = SHADOW_MAP_DEFAULT_SMSR_STEPS;
    q.smsrEpsilon = SHADOW_MAP_DEFAULT_SMSR_EPSILON;
    q.analyticEdge = SHADOW_MAP_DEFAULT_ANALYTIC_EDGE;
    q.analyticEdgeWidth = SHADOW_MAP_DEFAULT_ANALYTIC_EDGE_WIDTH;
    q.jitter = SHADOW_MAP_DEFAULT_JITTER;
    q.jitterTaps = SHADOW_MAP_DEFAULT_JITTER_TAPS;
    q.jitterRadius = SHADOW_MAP_DEFAULT_JITTER_RADIUS;
    q.layout = SHADOW_MAP_DEFAULT_LAYOUT;
    q.staticCache = SHADOW_MAP_DEFAULT_STATIC_CACHE;
    q.clipmapLevels = SHADOW_MAP_DEFAULT_CLIPMAP_LEVELS;
    q.clipmapBase = SHADOW_MAP_DEFAULT_CLIPMAP_BASE;
    q.clipmapResolution = SHADOW_MAP_DEFAULT_CLIPMAP_RESOLUTION;
    q.edgeHarden = SHADOW_MAP_DEFAULT_EDGE_HARDEN;
    q.edgeHardness = SHADOW_MAP_DEFAULT_EDGE_HARDNESS;
    q.edgeThreshold = SHADOW_MAP_DEFAULT_EDGE_THRESHOLD;
    q.ladderMode = SHADOW_MAP_DEFAULT_LADDER_MODE;
    q.ladderLambda = SHADOW_MAP_DEFAULT_LADDER_LAMBDA;
    q.ladderNear = SHADOW_MAP_DEFAULT_LADDER_NEAR;
    q.acne.enabled = SHADOW_MAP_DEFAULT_ACNE_ENABLED;
    q.acne.normalOffset = SHADOW_MAP_DEFAULT_ACNE_NORMAL_OFFSET;
    q.acne.normalTexels = SHADOW_MAP_DEFAULT_ACNE_NORMAL_TEXELS;
    q.acne.slopeScaled = SHADOW_MAP_DEFAULT_ACNE_SLOPE_SCALED;
    q.acne.slopeMax = SHADOW_MAP_DEFAULT_ACNE_SLOPE_MAX;
    return q;
}

// Clamp every field into the range the framework will honour. Called by the framework on arrival rather
// than trusted from the application, and safe to call on the application side too.
static inline void ShadowMapQualityClamp(ShadowMapQuality* q) {
    if (q == 0) {
        return;
    }
#define SHADOW_MAP_CLAMP_(v, lo, hi) ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))
    q->smsr = q->smsr ? 1 : 0;
    q->smsrMaxSteps = SHADOW_MAP_CLAMP_(q->smsrMaxSteps, 1, SHADOW_MAP_MAX_SMSR_STEPS);
    // Reject NaN as well as out-of-range values from configuration files.
    if (!(q->smsrEpsilon >= 0.0f && q->smsrEpsilon <= SHADOW_MAP_MAX_SMSR_EPSILON)) {
        q->smsrEpsilon = SHADOW_MAP_DEFAULT_SMSR_EPSILON;
    }
    q->analyticEdge = q->analyticEdge ? 1 : 0;
    q->analyticEdgeWidth = SHADOW_MAP_CLAMP_(q->analyticEdgeWidth, 0.25f, SHADOW_MAP_MAX_ANALYTIC_EDGE_WIDTH);
    q->jitter = q->jitter ? 1 : 0;
    q->jitterTaps = SHADOW_MAP_CLAMP_(q->jitterTaps, 1, SHADOW_MAP_MAX_JITTER_TAPS);
    q->jitterRadius = SHADOW_MAP_CLAMP_(q->jitterRadius, 0.0f, SHADOW_MAP_MAX_JITTER_RADIUS);
    q->layout = SHADOW_MAP_CLAMP_(q->layout, 0, SHADOW_MAP_LAYOUT_MAX);
    q->staticCache = q->staticCache ? 1 : 0;
    q->clipmapLevels = SHADOW_MAP_CLAMP_(q->clipmapLevels, 1, SHADOW_MAP_MAX_CLIPMAP_LEVELS);
    q->clipmapBase =
        SHADOW_MAP_CLAMP_(q->clipmapBase, SHADOW_MAP_MIN_CLIPMAP_BASE, SHADOW_MAP_MAX_CLIPMAP_BASE);
    q->clipmapResolution =
        SHADOW_MAP_CLAMP_(q->clipmapResolution, SHADOW_MAP_MIN_RESOLUTION, SHADOW_MAP_MAX_RESOLUTION);
    q->edgeHarden = q->edgeHarden ? 1 : 0;
    q->edgeHardness = SHADOW_MAP_CLAMP_(q->edgeHardness, 0.0f, 1.0f);
    q->edgeThreshold = SHADOW_MAP_CLAMP_(q->edgeThreshold, 0.05f, 0.95f);
    q->ladderMode = SHADOW_MAP_CLAMP_(q->ladderMode, 0, SHADOW_MAP_LADDER_MAX);
    q->ladderLambda = SHADOW_MAP_CLAMP_(q->ladderLambda, 0.0f, 1.0f);
    q->ladderNear = SHADOW_MAP_CLAMP_(q->ladderNear, 1.0f, 1000.0f);
    q->acne.enabled = q->acne.enabled ? 1 : 0;
    q->acne.normalOffset = q->acne.normalOffset ? 1 : 0;
    q->acne.normalTexels = SHADOW_MAP_CLAMP_(q->acne.normalTexels, 0.0f, SHADOW_MAP_MAX_ACNE_NORMAL_TEXELS);
    q->acne.slopeScaled = q->acne.slopeScaled ? 1 : 0;
    q->acne.slopeMax = SHADOW_MAP_CLAMP_(q->acne.slopeMax, 1.0f, SHADOW_MAP_MAX_ACNE_SLOPE_MAX);
#undef SHADOW_MAP_CLAMP_
}

#endif // FAST_SHADOW_MAP_H
