#ifndef FAST_WATER_H
#define FAST_WATER_H

// SOH [Enhancement] Breath of the Wild-style water.
//
// This header is the POLICY layer for the water material, in the same spirit as fast/shadow_map.h: the
// renderer, the backends and the game code all include it, so a number that describes how the water should
// look is written down exactly once and everything that reads it agrees by construction.
//
// -------------------------------------------------------------------------------------------------------
// What the feature is
// -------------------------------------------------------------------------------------------------------
// The N64 draws water as a flat polygon with a scrolling texture, vertex colour and alpha blending. This
// replaces that surface with a material whose colour comes from how much water the view ray travels through,
// which reflects the sky and the scene, refracts what is submerged, glints toward the sun, and foams where it
// meets anything solid.
//
// The implementation is staged (F0..F18 in the design document). Each stage leaves the game playable and is
// individually switchable, so the constants below are grouped by the stage that introduces them and stages
// that do not exist yet simply have no constants.
//
// -------------------------------------------------------------------------------------------------------
// The three signals
// -------------------------------------------------------------------------------------------------------
// Nearly every layer of the material is a function of one of three things, and the infrastructure exists to
// produce them well:
//
//   THICKNESS       how far the view ray travels inside the water, i.e. the distance from the water surface
//                   to whatever is behind it, MEASURED ALONG THE VIEW RAY and not vertically. This is why the
//                   linear depth target below stores ray distance from the eye rather than view-space Z or a
//                   normalised depth: a difference of two ray distances is exactly the thickness, with no
//                   reconstruction and no near/far algebra in the material.
//   NORMAL          the local tilt of the surface. Reflection, refraction, Fresnel, specular and glint are
//                   all functions of it; half a degree of error there changes the image more than a large
//                   error in the water's colour.
//   SUN VISIBILITY  whether the point is lit. The cascaded shadow map already in the fork is the source, and
//                   the rule for what it may darken is written down in the design document: the specular,
//                   the glint, the caustics and the foam, partially the scattering colour, and NEVER the
//                   reflection -- the sky is still there whether or not a tree shades that patch of water.
//
// -------------------------------------------------------------------------------------------------------
// Backend support
// -------------------------------------------------------------------------------------------------------
// Only Direct3D 11 implements the passes, exactly as with the shadow map. Every entry point on
// GfxRenderingAPI is a no-op by default and SupportsWater() answers false, so a backend that has not
// implemented them renders the original N64 water and nothing else changes. The application MUST consult
// SupportsWater() before suppressing the original draw, or an unsupported backend ends up with no water at
// all rather than with the water it started with.

// ---------------------------------------------------------------------------------------------------------
// Quality levels (design document 4.5). Each level is a superset of the one below it. The names are the
// feature's own; the menu maps a combobox onto these integers.
// ---------------------------------------------------------------------------------------------------------
#define WATER_QUALITY_OFF 0    // original N64 water; the feature costs nothing and touches nothing
#define WATER_QUALITY_LOW 1    // depth colour + Fresnel + sky; no refraction, no waves
#define WATER_QUALITY_MEDIUM 2 // + refraction, Gerstner waves, screen-space reflection at reduced steps
#define WATER_QUALITY_HIGH 3   // + full reflection steps, caustics, dedicated shadowing
#define WATER_QUALITY_COUNT 4

#define WATER_DEFAULT_QUALITY WATER_QUALITY_MEDIUM

// ---------------------------------------------------------------------------------------------------------
// F0 -- scene capture.
// ---------------------------------------------------------------------------------------------------------

// Mip levels generated for the scene-colour copy. The chain exists so that deep water can blur what shows
// through it and rough water can blur what it reflects, both of which are then selected by sampling a higher
// mip rather than by running a blur pass. Four levels reach 1/8 scale, which is as diffuse as either use
// needs.
//
// Mip 0 is full resolution rather than half, and that is deliberate: the one place the copy is read almost
// undistorted is the shoreline, where the water is thin and the refraction offset goes to zero, and a
// half-resolution base shows up there as a soft doubled edge against the sand. The chain provides every
// reduced scale anyway, so nothing is gained by starting lower -- only the sharp case is lost.
#define WATER_SCENE_COLOR_MIPS 4

// Linear depth is stored at FULL resolution and in 32-bit float, and both halves of that matter. The
// thickness signal drives the foam line and the shoreline colour ramp, and those are read at pixel accuracy
// where the water meets the sand; a half-resolution or 16-bit depth turns the foam line into a stair-stepped
// band that crawls as the camera moves. OoT's far plane is very distant, which squeezes hardware depth
// precision badly near the horizon, so the resolve linearises to world units up front and everything
// downstream reads a number it can subtract without thinking about the projection.
#define WATER_LINEAR_DEPTH_FULL_RES 1

// Value written to the linear depth target where the depth buffer holds the far plane, i.e. where nothing was
// drawn and the sky shows through. Water in front of it must read "infinitely thick" rather than "zero
// thickness", which is the difference between a deep blue horizon and a bright band across it (design
// document 2.4.4). Any value past the far plane works; this one is chosen to be obviously synthetic in the
// debug view.
#define WATER_DEPTH_SKY_SENTINEL 1.0e9f

// ---------------------------------------------------------------------------------------------------------
// F1 -- identifying which triangles are a water surface.
//
// The design document assumes the fork already answers this and only needs the answer routed differently.
// It does not: what exists is the COLLISION side (the game's water boxes, which the collision viewer draws
// and WaterBox_GetSurface1 queries), and nothing at all connecting that to the draw calls. So the bridge is
// built here, and it is built from the collision data rather than from render state.
//
// That choice is the one that keeps lava out. A render-state heuristic -- "translucent, scrolling texture,
// blue-ish vertex colour" -- describes the Fire Temple's lava as exactly as it describes Lake Hylia, and a
// false positive there is the worst failure this feature can have (design document 7, first row). A water
// box is authored data that says "the player swims here"; lava has none.
//
// The test per triangle is the conjunction of three things, and each one is load-bearing:
//   1. all three vertices lie within a box's XZ rectangle (grown by WATER_BOX_XZ_MARGIN),
//   2. all three sit within WATER_SURFACE_TOLERANCE of that box's surface height,
//   3. the triangle is close to horizontal.
// The third is what separates the lake's surface from the waterfall pouring into it and from the glass-like
// side walls of a water column -- both of which are inside the box and at the wrong orientation. They are
// F15's problem, not this one's.
// ---------------------------------------------------------------------------------------------------------

// How far, in world units, a vertex may sit from the box's stated surface height and still count. Generous
// on purpose: the authored water polygon and the authored collision box are two separate pieces of data and
// they do not agree to the unit, and some scenes tilt the drawn surface slightly. Too tight loses patches of
// a lake; too loose starts claiming the lake bed, which is why it is nothing like the box's depth.
#define WATER_SURFACE_TOLERANCE 12.0f

// How far outside the box's XZ rectangle a vertex may sit. Drawn water routinely overshoots its collision
// box by a little at the shoreline, and a hard edge there would leave a rim of original water around every
// lake.
#define WATER_BOX_XZ_MARGIN 24.0f

// Minimum |normal.y| for the triangle to count as a surface rather than a wall or a fall. cos(35 degrees);
// real lake surfaces are authored flat, so this only has to be loose enough for the slight tilts some
// scenes use.
#define WATER_MIN_SURFACE_NORMAL_Y 0.819f

// Ceiling on how many boxes one frame may carry. Scenes hold a handful; the cap exists so a malformed
// collision header costs a missing box rather than an unbounded copy.
#define WATER_MAX_BOXES 64

// ---------------------------------------------------------------------------------------------------------
// F3 -- thickness, absorption and scattering. Design document 2.4 and Annex A.
//
// Absorption is stated as HALF-LIGHT DISTANCES, in world units: how far light of that channel travels before
// half of it is gone. The shader wants an extinction coefficient, which is ln(2) divided by these -- the
// conversion happens once on upload rather than in the material.
//
// Per channel and exponential rather than a lerp between two colours, because that is what makes water read
// as a volume: red dies first, then green, blue travels furthest, so one curve gives the lake bed in
// near-natural colour at the shore, a green-turquoise in the middle distance and a deep petrol blue where the
// column is long. Two interpolated colours cannot produce that progression at all.
//
// Link is roughly 60-70 units tall, so 120 units of red is about two Links of water.
// ---------------------------------------------------------------------------------------------------------
#define WATER_DEFAULT_HALF_LIGHT_R 120.0f
#define WATER_DEFAULT_HALF_LIGHT_G 400.0f
#define WATER_DEFAULT_HALF_LIGHT_B 900.0f

// The colour water returns by scattering, and the thickness over which it reaches full strength. Without
// this term absorption alone drives deep water to black; with it, deep water is the milky turquoise the
// design describes.
#define WATER_DEFAULT_SCATTER_R 0.06f
#define WATER_DEFAULT_SCATTER_G 0.32f
#define WATER_DEFAULT_SCATTER_B 0.34f
#define WATER_DEFAULT_SCATTER_SATURATION 600.0f

// Gain on the original surface's own luminance before it modulates the scattering. The N64 water polygon is
// drawn with the room's vertex colours and environment tint, so its brightness already carries time of day
// and cave darkness; this scales that reading into a sensible 0..1 multiplier. Above 1 because the original
// water is usually a fairly dark blue.
#define WATER_DEFAULT_AMBIENT_GAIN 2.5f

// Gain applied to a claimed surface's OWN alpha to decide how much of it the material may replace.
//
// The identification is geometric, so anything flat and translucent at the water's height is claimed --
// including the ripples around a swimming player and the mist at the foot of a waterfall, both of which
// genuinely are at the water surface. What separates them from water is that they barely cover anything,
// and the original draw states that in its alpha. Composing over it turns a misidentification into a faint
// tint instead of a solid slab, for surfaces not yet discovered as well as the two that were.
//
// The gain exists because water itself is drawn semi-transparent: at this value anything at roughly half
// alpha or above takes the material in full, while a near-transparent haze keeps almost all of its own
// appearance.
#define WATER_DEFAULT_COVERAGE_GAIN 2.2f

// Thickness over which the surface fades in at the very shoreline, so the waterline meets the sand as a soft
// edge rather than a cut (design document 2.19, item 11).
#define WATER_DEFAULT_SHORE_FADE 18.0f

// ---------------------------------------------------------------------------------------------------------
// Debug views (F0/F1). The menu exposes these as a combobox, the way the shadow map's cascade view is.
// ---------------------------------------------------------------------------------------------------------
#define WATER_DEBUG_OFF 0
#define WATER_DEBUG_SCENE_COLOR 1  // the captured colour copy, drawn in the corner
#define WATER_DEBUG_LINEAR_DEPTH 2 // linear depth as greyscale, scaled by WATER_DEBUG_DEPTH_SCALE
#define WATER_DEBUG_BOTH 3         // both thumbnails
// F1: skip drawing every triangle identified as a water surface. The surface vanishing is what proves BOTH
// halves at once -- that the identification found it, and that the suppression can take it out of the frame
// cleanly for the new material to be put in its place. A magenta tint would only have proved the first, and
// would have needed a shader to say it.
#define WATER_DEBUG_HIDE_SURFACES 4
#define WATER_DEBUG_COUNT 5

// World distance that maps to white in the greyscale depth view. Hyrule Field is a few thousand units across,
// so this shows the whole visible range without the near field being crushed to black.
#define WATER_DEBUG_DEPTH_SCALE 4000.0f

// Fraction of the screen width one debug thumbnail occupies.
#define WATER_DEBUG_THUMB_FRACTION 0.25f

#ifdef __cplusplus

#include <stddef.h>
#include <stdint.h>

namespace Fast {

// One of the scene's water boxes, as the game knows it: an axis-aligned XZ rectangle with a surface height.
//
// Pushed by the game each frame rather than read by the renderer, because deciding WHICH boxes are active is
// game knowledge that the renderer has no business duplicating -- the room filter packed into the box's
// property bits, the hard-coded extra box Zora's Domain needs so the player can pass under the waterfall, and
// the guard for a collision header that has not finished loading. All three live in z_bgcheck.c and all
// three would have been silently wrong here.
struct WaterBoxDesc {
    float xMin = 0.0f;
    float zMin = 0.0f;
    float xLength = 0.0f;
    float zLength = 0.0f;
    float ySurface = 0.0f;
    // Stable across frames for the same body of water, so a mesh cache (F2) and an appearance profile (F16)
    // can be keyed by it. Scene number in the high bits, box index in the low ones.
    uint32_t id = 0;
};

// Everything the water material needs to know about the camera and the frame, gathered once and pushed to
// the backend in one call.
//
// A struct rather than a long argument list because SetShadowMapParams taught the lesson: it grew to ten
// arguments and is invoked from half a dozen early-exit paths, every one of which has to be edited whenever
// one more number is needed. This is passed by const reference and grows without touching a single call site.
struct WaterFrameParams {
    // World -> clip, row-major, the same convention the interpreter uses for its own matrices.
    float viewProj[16] = {};
    // Its inverse. Used to turn a screen pixel plus a depth value back into a world position, which is how
    // the linear depth resolve works and how the caustics will project onto the floor later.
    float invViewProj[16] = {};

    // Camera position in world space, recovered by intersecting two unprojected view rays rather than by
    // assuming the near-plane centre is the eye. The near plane is close enough to the eye that the
    // approximation is harmless for cascade fitting, but thickness is a DIFFERENCE of distances from this
    // point, and a systematic error in it biases the shoreline everywhere at once.
    float camPos[3] = { 0.0f, 0.0f, 0.0f };
    // Unit vector along the view axis.
    float camDir[3] = { 0.0f, 0.0f, 1.0f };

    // Recovered from the projection, in world units. Carried for the debug readout and for the backends that
    // need them to build their own depth transforms; the material itself never uses them, by design.
    float nearPlane = 0.0f;
    float farPlane = 0.0f;

    // Matrix NDC x per screen NDC x. The screen is WIDER than the projection matrix says: every transformed
    // vertex has its clip x divided by the aspect ratio so that a 4:3 projection fills a widescreen window,
    // which means the edge of the screen is not at matrix NDC +/-1 but further out, by exactly the
    // reciprocal of that division. Anything that goes the other way -- turning a PIXEL back into a world
    // position, which is what the depth resolve does -- has to undo it, or every reconstructed position is
    // wrong horizontally and the error grows toward the screen edges. Same correction the cascade fit makes
    // when it measures the frustum; derived from the same function so the two cannot disagree.
    float ndcXScale = 1.0f;

    // Seconds since the feature started running, monotonic. Drives every scroll and phase in the material.
    // Held here rather than read from a clock inside the shader path so that all layers of one frame share a
    // single value and cannot drift apart from each other.
    float time = 0.0f;

    // Dimensions of the target the frame is being rendered into, in pixels.
    int screenWidth = 0;
    int screenHeight = 0;

    // How much of a claimed surface the material may replace, as a gain on that surface's own alpha (see
    // WATER_DEFAULT_COVERAGE_GAIN). Exposed as a slider rather than fixed, because the quantity it has to be
    // set against -- the alpha the combiner actually produces for a water surface -- is not something the
    // renderer can read back, and guessing it a build at a time is the slow way to find one number.
    float coverageGain = WATER_DEFAULT_COVERAGE_GAIN;

    // Quality level in force this frame (WATER_QUALITY_*). Pushed with the rest so the backend can size its
    // resources and skip work without reaching back into the application's config.
    int quality = WATER_DEFAULT_QUALITY;

    // WATER_DEBUG_*.
    int debugView = WATER_DEBUG_OFF;
};

} // namespace Fast

#endif // __cplusplus

#endif // FAST_WATER_H
