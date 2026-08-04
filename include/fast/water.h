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
// Debug views (F0). The menu exposes these as a slider, the way the shadow map's cascade view is exposed.
// ---------------------------------------------------------------------------------------------------------
#define WATER_DEBUG_OFF 0
#define WATER_DEBUG_SCENE_COLOR 1 // the captured colour copy, drawn in the corner
#define WATER_DEBUG_LINEAR_DEPTH 2 // linear depth as greyscale, scaled by WATER_DEBUG_DEPTH_SCALE
#define WATER_DEBUG_BOTH 3         // both thumbnails plus the numeric near/far readout
#define WATER_DEBUG_COUNT 4

// World distance that maps to white in the greyscale depth view. Hyrule Field is a few thousand units across,
// so this shows the whole visible range without the near field being crushed to black.
#define WATER_DEBUG_DEPTH_SCALE 4000.0f

// Fraction of the screen width one debug thumbnail occupies.
#define WATER_DEBUG_THUMB_FRACTION 0.25f

#ifdef __cplusplus

#include <stddef.h>

namespace Fast {

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

    // Quality level in force this frame (WATER_QUALITY_*). Pushed with the rest so the backend can size its
    // resources and skip work without reaching back into the application's config.
    int quality = WATER_DEFAULT_QUALITY;

    // WATER_DEBUG_*.
    int debugView = WATER_DEBUG_OFF;
};

} // namespace Fast

#endif // __cplusplus

#endif // FAST_WATER_H
