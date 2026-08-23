@prism(type='hlsl', name='Fast3D HLSL Shader', version='1.0.0', description='Ported shader to prism', author='Emill & Prism Team')

@if(o_root_signature)
    @if(o_textures[0])
        #define RS "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | DENY_VERTEX_SHADER_ROOT_ACCESS), CBV(b0, visibility = SHADER_VISIBILITY_PIXEL), DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL), DescriptorTable(Sampler(s0), visibility = SHADER_VISIBILITY_PIXEL)"
    @end
    @if(o_textures[1])
        #define RS "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | DENY_VERTEX_SHADER_ROOT_ACCESS), CBV(b0, visibility = SHADER_VISIBILITY_PIXEL), DescriptorTable(SRV(t1), visibility = SHADER_VISIBILITY_PIXEL), DescriptorTable(Sampler(s1), visibility = SHADER_VISIBILITY_PIXEL)"
    @end
@end

struct PSInput {
    float4 position : SV_POSITION;
@for(i in 0..2)
    @if(o_textures[i])
        float2 uv@{i} : TEXCOORD@{i};
        @{update_floats(2)}
        @for(j in 0..2)
            @if(o_clamp[i][j])
                @if(j == 0)
                    float texClampS@{i} : TEXCLAMPS@{i};
                @else
                    float texClampT@{i} : TEXCLAMPT@{i};
                @end
                @{update_floats(1)}
            @end
        @end
    @end
@end

@if(o_fog)
float4 fog : FOG;
@{update_floats(4)}
@end
@if(o_grayscale)
float4 grayscale : GRAYSCALE;
@{update_floats(4)}
@end
@if(o_toon || o_shadow_map)
float3 normal : NORMAL;
@{update_floats(3)}
@end
@if(o_shadow_map)
// xyz is the world position; w is 1 for scenery and 0 for a character (see the note in ShaderOpts).
float4 worldPos : WORLDPOS;
@{update_floats(4)}
@end

@for(i in 0..o_inputs)
    @if(o_alpha)
        float4 input@{i + 1} : INPUT@{i};
        @{update_packed_color()}
    @else
        float3 input@{i + 1} : INPUT@{i};
        @{update_packed_color()}
    @end
@end
};

@if(o_textures[0]) 
    Texture2D g_texture0 : register(t0);
    SamplerState g_sampler0 : register(s0);
@end
@if(o_textures[1]) 
    Texture2D g_texture1 : register(t1);
    SamplerState g_sampler1 : register(s1);
@end

@if(o_masks[0]) Texture2D g_textureMask0 : register(t2);
@if(o_masks[1]) Texture2D g_textureMask1 : register(t3);

@if(o_blend[0]) Texture2D g_textureBlend0 : register(t4);
@if(o_blend[1]) Texture2D g_textureBlend1 : register(t5);

cbuffer PerFrameCB : register(b0) {
    uint noise_frame;
    float noise_scale;
}

// SOH [Enhancement] Toon lighting. Its own cbuffer (b2 — the first free slot at this LUS base) so
// PerFrameCB stays frame-global; only the toon pixel shader declares/reads it. Layout matches the
// PerToonCB C++ struct; each float3 followed by a float fills a 16-byte register per HLSL packing.
@if(o_toon)
cbuffer PerToonCB : register(b2) {
    float3 toon_light_dir;
    float toon_ramp_center;
    float3 toon_light_color;
    float toon_ramp_softness;
    float3 toon_ambient;
    float toon_highlight_intensity;
    float toon_shadow_intensity;
    float toon_debug;
    float2 _toon_pad;
}
@end

// SOH [Enhancement] Cascaded shadow maps. The cascade array binds past the combiner's own texture slots
// (SHADER_MAX_TEXTURES) so it never collides with a texel sampler, and its cbuffer takes b3 (b0 per-frame,
// b1 per-draw, b2 toon). Only the receiver variant declares any of this.
@if(o_shadow_map)
// Declared <float> and read with a plain sampler, then compared in the shader, rather than through a
// SamplerComparisonState. Comparison sampling would fold a filtered 2x2 into every fetch for free; what it
// costs is the individual results, and the diagnostic views want the raw coverage the comparisons produced
// rather than a number the hardware already averaged. Gather (see SampleShadowPCF4) buys the fetch count
// back. The element type is explicit because the fetch has to come back as one float, not a float4 to be
// truncated.
Texture2DArray<float> g_shadowMap : register(t6);
SamplerState g_shadowSampler : register(s6);

// SOH [Enhancement] The ACTOR caster layer, which may live in an array of its own at its own resolution
// (see fast/shadow_map.h). While the two layers are the same size the backend binds the SAME view here, so
// every fetch below reads exactly the texels it read before the layers could be sized apart.
//
// Selected with a branch rather than by duplicating the kernel: `isActor` is uniform for the whole of a
// lookup, both textures are referenced statically, and no index is dynamic -- so this stays inside what
// ps_4_0 will map, which duplicating eighty lines of tuned filtering by hand would have risked getting
// subtly wrong instead.
Texture2DArray<float> g_shadowMapActors : register(t7);
SamplerState g_shadowActorSampler : register(s7);

// SOH [Enhancement] Filterable shadow maps (technique 1 -- see fast/shadow_map.h). Holds a quantity whose
// AVERAGE is meaningful -- exp(k*d) for ESM, the first two or four power moments for VSM and MSM -- so the
// MAP can be blurred and the receiver stays one bilinear fetch.
//
// Its own slot rather than reinterpreting the depth one: the formats differ and so does the filtering that
// is correct for each. Point is the only correct setting for a hand-rolled comparison; linear is the whole
// point of a filterable quantity.
//
// WORLD LAYER ONLY. The actor layer keeps depth and PCF -- it lives in a different texture when the two
// resolutions are split, and it is redrawn every frame, so it would pay the resolve and both blur axes
// every frame for the shadows this technique helps least. The backend allocates only the world cascades,
// so this array's slice index IS the cascade index.
Texture2DArray<float4> g_shadowMoments : register(t8);
SamplerState g_shadowMomentSampler : register(s8);

// SOH [Enhancement] Screen-space shadow mask (technique 5 -- see fast/shadow_map.h). R holds the shadow
// term resolved for the whole frame, G the view depth it was resolved at.
//
// G is what makes the mask safe to use. It is resolved from a prepass of the world CASTER geometry, which
// is not quite everything that receives -- and nothing at all of the alpha-blended surfaces, the water or
// the particles, which are not in it by construction. A receiver compares its own depth against G and uses
// the mask only where they agree; everywhere else it samples the cascades exactly as it did before.
Texture2D<float4> g_shadowMask : register(t9);
SamplerState g_shadowMaskSampler : register(s9);

// Everything is float4-shaped on purpose: HLSL gives each element of a `float arr[n]` its own 16-byte
// register, so a scalar array would waste three quarters of its space and make the C++ layout easy to get
// subtly wrong. Layout matches the PerShadowCB C++ struct exactly.
cbuffer PerShadowCB : register(b3) {
    row_major float4x4 shadow_view_proj[@{o_shadow_max_cascades}];
    float4 shadow_splits;      // far distance of each cascade, world units
    float4 shadow_texel_world; // world size of one texel, per cascade
    float4 shadow_texel_uv;    // one texel in UV terms (1/resolution), per cascade
    // x = active cascade count (0 = no shadow map this frame), y = cross-fade band as a fraction of the
    // cascade, z unused, w = darkness where fully occluded.
    float4 shadow_params;
    // x = the furthest view depth any cascade's footprint reaches. Past it no cascade covers anything, so
    //     the lookup is skipped outright. Not derivable from the splits: a cascade's box overshoots its own
    //     band by a long way, and cutting at the split would delete real shadows.
    // y = debug view selector. Modes 1 and 2 show what the system produced; 3 to 5 and 7 show what it was
    //     given -- see the channel in PSMain for how to read them against each other.
    //     0 off
    //     1 everything OUTSIDE a cascade's footprint painted fully occluded instead of silently lit, which
    //       is the only way to see where a cascade actually ends
    //     2 the two caster layers by colour: green = occluded by the world layer, red = by the actor layer
    //     3 the receiver normal, as a colour
    //     4 where that normal came from: green = the draw's vertex normal (brightness = its interpolated
    //       length), red = a face normal recovered from screen derivatives
    //     5 the coverage the depth comparison produced, which is also what the shaded picture uses
    //     7 cascade index, red/green/blue from nearest to furthest
    //     (6, 8 and 9 read values that no longer exist -- they showed the receiver-plane gradient, the
    //      incidence taper and the anisotropic kernel's reach, all removed with the machinery they measured)
    // z, w = unused. Kept float4-shaped because HLSL gives each cbuffer element its own 16-byte register
    //     anyway, so shrinking this saves nothing and is one more chance to get the C++ layout wrong.
    float4 shadow_range;
    // World-space bounds of the ACTOR caster layer. Everything in that layer is inside this box, so a
    // receiver with no part of the box behind it along the light cannot be shadowed by it -- see the test in
    // ShadowLitLayers. An empty layer arrives inverted and fails every test, which is what "no characters"
    // should do.
    float4 shadow_actor_min;
    float4 shadow_actor_max;
    // One texel of the ACTOR layer in UV terms, per cascade. Equal to shadow_texel_uv while the two layers
    // share a resolution; separate once that layer is sized on its own (see fast/shadow_map.h).
    float4 shadow_actor_texel_uv;
    // SOH [Enhancement] Edge quality (see fast/shadow_map.h). Three registers holding the switches and
    // tuning for the techniques that shape the shadow's EDGE, as opposed to deciding where it falls.
    // Packed rather than named one per register because a cbuffer gives every scalar its own 16 bytes.
    //   edge:   x = analytic edge on/off, y = its ramp width in texels,
    //           z = jitter on/off,        w = jitter tap count
    //   jitter: x = jitter radius in texels, y = per-frame rotation offset (0 when not temporal),
    //           z = filter mode (SHADOW_MAP_FILTER_*), w = ESM exponent
    //   filter: x = bleed reduction, y = screen-space mask active, z = map blur radius in texels,
    //           w = unused
    float4 shadow_edge;
    float4 shadow_jitter;
    float4 shadow_filter;
    // SOH [Enhancement] Screen-space mask (technique 5). x = a mask was built this frame, yz = one screen
    // pixel in UV, w = how far the mask's stored depth may differ from this receiver's before the mask is
    // judged to describe a different surface.
    float4 shadow_mask;
    // SOH [Enhancement] Shadow acne (see fast/shadow_map.h). The magnitudes arrive already zeroed when
    // their switch is off, so the shader multiplies rather than branches.
    //   acne0: x = corrections enabled, y = normal offset in texels, z = light offset in world units,
    //          w = depth bias in world units along the light
    //   acne1: x = scale by how edge-on the surface is, y = the ceiling on that scale,
    //          z = apply in the ordinary receiver too, w = unused
    float4 shadow_acne0;
    float4 shadow_acne1;
}

// One depth fetch, compared by hand. The sampler filters point-wise on purpose: averaging stored depths
// and then comparing once is not the same thing as comparing per texel and averaging the results, and only
// the latter gives a correct penumbra.
// Receiver plane depth bias: compare against the depth the RECEIVER'S OWN PLANE would have at the texel
// being tapped, not the depth it has at the centre of the kernel.
//
// This is the root cause of acne under a wide filter, and every bias in this file up to now has been
// treating the symptom. A PCF tap reads stored depth one or more texels away from the sample point but
// compares it against the receiver's depth AT the sample point. On a surface tilted with respect to the
// light those two are not the same number, and the difference grows with both the tilt and the kernel
// width -- so the sixteen-tap filter was itself manufacturing the acne that the constant, slope and normal
// biases were then paying to hide, which is why they had to be so large and why they cost peter panning.
//
// `grad` is how fast the receiver's depth changes per unit of shadow-map uv, so this recovers the plane's
// own depth at the tap and compares like with like. Nothing is displaced: the correction is exact for a
// flat receiver and needs no margin, which is what makes it free of panning.
// Four taps in a quincunx around the centre, written out rather than looped over a local array: a local
// array can land in an indexable temp register, which ps_4_0 refuses to map. The slice is cast explicitly
// -- it is a texture coordinate and has to arrive as a float, and leaving that implicit is what produces
// the truncation warnings.
// Bilinear PCF over the four texels surrounding the sample point: compare each, then weight the RESULTS by
// the sub-texel position. Comparing first and filtering after is the whole point -- filtering the stored
// depths and comparing once would give a wrong penumbra.
//
// The offsets have to be a full texel apart, and land on texel centres. An earlier version kept the
// half-texel quincunx offsets that suited a hardware comparison sampler, where every fetch already
// straddles four texels. Against a point sampler those four taps usually land inside the SAME texel,
// return the same value, and average to exactly one hard sample -- no filtering at all, which is what made
// edges stair-step.
// SOH [Enhancement] Analytic edge reconstruction (technique 2 -- see fast/shadow_map.h).
//
// Coverage from where the boundary actually CROSSES the quad, rather than from bilinearly blending four
// binary comparisons. Same four depths, no extra fetch.
//
// Let g = stored - z, the signed slack at each corner: positive where that corner does not occlude. The
// lit region is g >= 0, so the boundary is the zero contour of g, and bilinear g is a good model of it
// inside one quad. Interpolating g and taking its gradient gives the signed distance from the sample point
// to that contour, in texels -- and a ramp on the distance is an edge whose position varies continuously
// with the receiver instead of snapping to the grid. That snapping is the staircase.
//
// Exact for one straight boundary through the quad, which is walls, steps, roofs and platform edges. Where
// the quad holds more than one boundary the gradient is meaningless, but so is the bilinear blend, and the
// magnitude guard below returns the hard answer rather than an invented soft one.
//
// `width` widens the ramp past its geometric one-texel extent, which is the cheapest softening in the
// system: arithmetic, no fetches, no bandwidth.
float ShadowAnalyticCoverage(float4 stored, float z, float2 subTexel, float width) {
    // Gather's component order: w is (0,0), z is (1,0), x is (0,1), y is (1,1).
    float4 g = stored - z;
    float row0 = lerp(g.w, g.z, subTexel.x); // v = 0
    float row1 = lerp(g.x, g.y, subTexel.x); // v = 1
    float value = lerp(row0, row1, subTexel.y);
    // Gradient in TEXEL units, which is what makes the distance below a texel count.
    float du = lerp(g.z - g.w, g.y - g.x, subTexel.y);
    float dv = row1 - row0;
    float gradient = length(float2(du, dv));
    // A flat quad has no boundary in it and no gradient to divide by. Falling back to the hard comparison
    // is right: there is genuinely nothing to antialias, and every neighbouring quad that DOES hold the
    // boundary is producing the ramp.
    if (gradient < 1e-7) {
        return step(0.0, value);
    }
    // Signed distance to the contour, in texels, then a linear ramp of `width` texels centred on it.
    return saturate(0.5 + ((value / gradient) / max(width, 1e-4)));
}

// SOH [Enhancement] Interleaved gradient noise (technique 3 -- see fast/shadow_map.h).
//
// One hash of the pixel coordinate, returning 0..1. Chosen over a texture lookup because it costs no
// bandwidth and no bind, and over a plain hash because its output is spatially well distributed at the
// scale of a few pixels -- which is exactly the scale a rotated tap pattern is trying to decorrelate over.
float ShadowJitterNoise(float2 pixel) {
    return frac(52.9829189 * frac(dot(pixel, float2(0.06711056, 0.00583715))));
}

float SampleShadowPCF4(float2 uv, float z, float slice, float texelUv, bool isActor) {
    // Position in texel space, offset so flooring lands on the lower-left of the surrounding quad.
    float2 texelPos = uv / texelUv - 0.5;
    float2 baseTexel = floor(texelPos);
    float2 subTexel = texelPos - baseTexel;
    float2 uv00 = (baseTexel + 0.5) * texelUv;

    // Component order below is Gather's own, and the fetch-by-fetch path is written to match it: w is the
    // texel at (0,0) from uv00, z is (1,0), x is (0,1), y is (1,1). Holding all four as one float4 is the
    // point of this function's shape -- the compare and the blend are then single vector instructions
    // instead of four scalar comparisons and three lerps, and the kernel runs this sixteen times.
@if(o_shadow_gather)
    // One instruction for all four depths. Gather returns exactly the 2x2 footprint bilinear filtering
    // would have used, so this reads the same four texels the four fetches below read -- and every texel is
    // still compared on its own, against its own point on the receiver plane, and still weighted by hand.
    // The output is identical; only the number of texture instructions changes, sixteen to four across the
    // whole kernel.
    //
    // Sampled at the CENTRE of the quad rather than at a texel, deliberately. The footprint the hardware
    // picks is decided in fixed point, and asking at a texel centre puts that decision half a texel from a
    // boundary in each direction -- orders of magnitude more margin than the hardware's sub-texel precision
    // -- so the four texels are the computed ones and not their neighbours.
    float4 stored = isActor ? g_shadowMapActors.Gather(g_shadowActorSampler, float3(uv00 + texelUv * 0.5, slice))
                            : g_shadowMap.Gather(g_shadowSampler, float3(uv00 + texelUv * 0.5, slice));
@else
    float4 stored;
    if (isActor) {
        stored.w = g_shadowMapActors.SampleLevel(g_shadowActorSampler, float3(uv00, slice), 0);
        stored.z = g_shadowMapActors.SampleLevel(g_shadowActorSampler, float3(uv00 + float2(texelUv, 0.0), slice), 0);
        stored.x = g_shadowMapActors.SampleLevel(g_shadowActorSampler, float3(uv00 + float2(0.0, texelUv), slice), 0);
        stored.y =
            g_shadowMapActors.SampleLevel(g_shadowActorSampler, float3(uv00 + float2(texelUv, texelUv), slice), 0);
    } else {
        stored.w = g_shadowMap.SampleLevel(g_shadowSampler, float3(uv00, slice), 0);
        stored.z = g_shadowMap.SampleLevel(g_shadowSampler, float3(uv00 + float2(texelUv, 0.0), slice), 0);
        stored.x = g_shadowMap.SampleLevel(g_shadowSampler, float3(uv00 + float2(0.0, texelUv), slice), 0);
        stored.y = g_shadowMap.SampleLevel(g_shadowSampler, float3(uv00 + float2(texelUv, texelUv), slice), 0);
    }
@end

    // SOH [Enhancement] Technique 2 (see fast/shadow_map.h) consumes the SAME four depths and reconstructs
    // where the boundary crosses the quad, rather than blending the four comparisons. Branching on a
    // uniform, so a draw takes one path or the other and neither pays for the one it skipped.
    if (shadow_edge.x > 0.5) {
        return ShadowAnalyticCoverage(stored, z, subTexel, shadow_edge.y);
    }

    // step(a, b) is b >= a, so this is "the receiver is at or in front of the stored depth" -- 1 where the
    // texel does not occlude -- for all four at once. One reference depth for the whole quad: the offset
    // that keeps a surface from shadowing itself is applied by the rasterizer during the depth pass, not
    // here (see SHADOW_MAP_SLOPE_BIAS).
    float4 lit = step(z, stored);

    // Bilinear weights, written out. lerp(lerp(w, z, sx), lerp(x, y, sx), sy) is exactly this sum, and as a
    // dot it is one instruction instead of three dependent ones. Comparing first and filtering after is the
    // whole point of the kernel -- filtering the stored depths and comparing once would give a wrong
    // penumbra -- and that ordering is unchanged: `lit` is already the comparison result.
    float2 inv = 1.0 - subTexel;
    float4 weights = float4(inv.x * subTexel.y, subTexel.x * subTexel.y, subTexel.x * inv.y, inv.x * inv.y);
    return dot(lit, weights);
}

// SOH [Enhancement] Stochastic jitter (technique 3 -- see fast/shadow_map.h).
//
// Spread the taps over a disk and rotate that disk by a per-pixel angle. The step between texels does not
// shrink, but neighbouring pixels no longer step at the same place, so the boundary reads as dither rather
// than as a staircase.
//
// A Vogel (golden-angle) spiral rather than a grid: it has no preferred axis, so there is no direction
// along which the pattern itself can print -- which is the failure mode of a rotated square kernel, and is
// what turned an earlier widened filter into visible wedges.
//
// Disabled is a uniform branch straight to the single quad, so a draw with jitter off is byte for byte the
// cost it was. [loop] and not [unroll] for the reason documented on the layer loop below: this file is
// compiled by FXC inside the frame a material first draws, and an unrolled sixteen-tap body is that hitch.
float SampleShadowJittered(float2 uv, float z, float slice, float texelUv, bool isActor, float2 pixel) {
    if (shadow_edge.z < 0.5) {
        return SampleShadowPCF4(uv, z, slice, texelUv, isActor);
    }

    uint taps = (uint)max(shadow_edge.w, 1.0);
    // The frame term is zero unless temporal jitter is on, in which case the pattern advances and the grain
    // moves instead of standing still as a fixed texture over the scene.
    float angle = (ShadowJitterNoise(pixel) + shadow_jitter.y) * 6.28318530718;
    float2 rot = float2(cos(angle), sin(angle));
    float radius = shadow_jitter.x * texelUv;

    float sum = 0.0;
    [loop]
    for (uint i = 0; i < taps; i++) {
        // sqrt of the index fraction distributes the taps by AREA, so the disk is evenly covered rather
        // than crowded at the centre. 2.3999632 is the golden angle in radians.
        float r = sqrt(((float)i + 0.5) / (float)taps);
        float theta = (float)i * 2.3999632;
        float2 unit = float2(cos(theta), sin(theta));
        // Complex multiply: rotate the spiral's own direction by this pixel's angle.
        float2 dir = float2((unit.x * rot.x) - (unit.y * rot.y), (unit.x * rot.y) + (unit.y * rot.x));
        sum += SampleShadowPCF4(uv + (dir * (r * radius)), z, slice, texelUv, isActor);
    }
    return sum / (float)taps;
}

// SOH [Enhancement] Recover the lit fraction from stored moments (technique 1 -- see fast/shadow_map.h).
//
// One bilinear fetch, already averaged by the blur and by the hardware, turned back into coverage. This is
// where a filterable map pays off: the softness came from a blur over the map, so nothing here loops.
float ShadowMomentLit(float4 m, float z, int mode, float exponent, float bleed) {
    if (mode == 1) {
        // ESM. The stored average of exp(k*d) against exp(k*z) for this receiver. Saturated because the
        // estimate runs above one wherever the blur averaged in something nearer than this receiver -- an
        // overshoot, not an occlusion.
        return saturate(exp(-exponent * z) * m.x);
    }

    if (mode == 2) {
        // VSM. Chebyshev's inequality on the first two moments bounds the lit fraction from above.
        float mean = m.x;
        float variance = max(m.y - (mean * mean), 1.0e-6);
        float diff = z - mean;
        float bound = variance / (variance + (diff * diff));
        // In front of the mean nothing can occlude, and the bound is not the answer there.
        float lit = (z <= mean) ? 1.0 : bound;
        // The bound is loose where two occluders at different depths share a texel, which is seen as a
        // shadow going translucent in its middle. Rescaling the low tail away is the standard remedy.
        return saturate((lit - bleed) / max(1.0 - bleed, 1.0e-4));
    }

    // MSM, four power moments (Peters & Klein). Solves for the tightest bound consistent with all four,
    // which is what holds up under the overlapping occluders VSM bleeds through.
    //
    // Biased a hair towards the moments of a flat fully lit surface first: a texel whose casters are all at
    // one depth -- most of an empty map -- makes the system below exactly degenerate.
    float4 b = lerp(m, float4(0.0, 0.375, 0.0, 0.375), 6.0e-5);

    // Cholesky of the Hankel matrix the moments form, solved in place.
    float d22 = b.y - (b.x * b.x);
    float l32d22 = b.z - (b.x * b.y);
    float d33d22 = ((b.w - (b.y * b.y)) * d22) - (l32d22 * l32d22);
    float invD22 = 1.0 / max(d22, 1.0e-9);
    float l32 = l32d22 * invD22;

    float3 c;
    c.x = 1.0;
    c.y = z - b.x;
    c.z = (z * z) - b.y - (l32 * c.y);
    c.y *= invD22;
    c.z *= d22 / max(d33d22, 1.0e-9);
    c.y -= l32 * c.z;
    c.x -= dot(c.yz, b.xy);

    // Roots of c.z t^2 + c.y t + c.x, ordered.
    float safeC2 = (c.z < 0.0) ? min(c.z, -1.0e-9) : max(c.z, 1.0e-9);
    float p = c.y / safeC2;
    float q = c.x / safeC2;
    // Clamped at zero rather than trusted: a moment set the blur has pushed slightly out of validity gives
    // a negative discriminant, and a NaN here would spread through the whole shaded pixel.
    float root = sqrt(max((p * p * 0.25) - q, 0.0));
    float z1 = (-p * 0.5) - root;
    float z2 = (-p * 0.5) + root;

    float4 sw = (z2 < z) ? float4(z1, z, 1.0, 1.0) : ((z1 < z) ? float4(z, z1, 0.0, 1.0) : float4(0.0, 0.0, 0.0, 0.0));
    float denom = (z2 - sw.y) * (z - z1);
    float quotient = ((sw.x * z2) - (b.x * (sw.x + z2)) + b.y) / ((abs(denom) < 1.0e-9) ? 1.0e-9 : denom);
    float occluded = saturate(sw.z + (sw.w * quotient));
    return saturate(((1.0 - occluded) - bleed) / max(1.0 - bleed, 1.0e-4));
}

// Project into one cascade and return how lit that cascade says this point is (1 = lit, 0 = occluded).
// Outside the cascade's footprint there is nothing to occlude, so the answer is "lit" -- which is also
// what the border-clamped sampler returns, but checking here avoids the fetch entirely.
// NOTE ON INDEXING: ps_4_0 has no instruction for reading a vector component by a runtime value, so
// nothing below may write shadow_splits[c] or shadow_texel_uv[c] with a non-literal c. Doing so does not
// just fail -- the compiler first treats the expression as the whole float4, which shows up as
// "implicit truncation of vector type" warnings, and only then reports "cannot map expression to ps_4_0".
// Every access here is therefore a literal .x/.y/.z/.w behind a small selector.
// Single return from a pre-initialized local, not a chain of early returns. Early returns here drew
// "warning X4000: use of potentially uninitialized variable": the compiler flattens this control flow and
// then cannot prove every path assigned the result, and that unresolved flow is part of what it refuses to
// map to ps_4_0.
float ShadowSplitAt(uint c) {
    float s = shadow_splits.z;
    if (c == 0) {
        s = shadow_splits.x;
    } else if (c == 1) {
        s = shadow_splits.y;
    }
    return s;
}

// Compare one cascade. The per-cascade values arrive by value rather than being looked up, which is what
// keeps the caller's selection on literal indices (see the note above). `slice` is only ever a texture
// coordinate, and those may be dynamic.
// One cascade lookup, split into the projection and the fetches so the fetches can sit behind a real branch.
//
// The cross-fade partner's whole kernel used to be evaluated for every pixel and then multiplied by a blend
// weight that is zero outside a band covering a tenth of a cascade's range -- thrown away on the large
// majority of the screen. Splitting is what lets ShadowLit skip it instead.
struct ShadowProjection {
    float2 uv;
    float z; // ndc depth of the receiver
    float texelUv;
    float slice;  // texture-array slice, this layer's offset included
    float inside; // 1 where the cascade covers this point, 0 where there is nothing to sample
};

// The direction the light travels, which every cascade shares.
//
// Each cascade's matrix scales the light's unit z axis by its own 1/(zFar - zNear), so the third column IS
// that axis times a positive number -- and normalising throws that number away. All four therefore give the
// same vector, which is why it is taken from cascade 0 here and handed down, rather than recovered inside
// each projection from the matrix that projection happens to hold. A pixel used to pay for this twice over
// (once per cascade it looks at) plus once more in PSMain, for three identical results.
float3 ShadowLightAxis() {
    return normalize(shadow_view_proj[0]._13_23_33);
}

// Everything about a lookup except the fetches. No derivatives here, which is what makes this safe to call
// from inside a branch -- and the partner projection is now built only where it is actually read.
ShadowProjection ShadowProject(float3 p, float4x4 viewProj, float texelUv, uint cascade, float sliceBase) {
    float4 clip = mul(float4(p, 1.0), viewProj);

    float safeW = abs(clip.w) > 1e-6 ? clip.w : 1e-6;
    float3 ndc = clip.xyz / safeW;
    // NDC -> texture space (y flips: NDC is +up, textures are +down).
    float2 uv = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    ShadowProjection o;
    o.uv = uv;
    o.z = ndc.z;
    o.texelUv = texelUv;
    o.slice = sliceBase + (float)cascade;
    o.inside = (clip.w > 0.0 && all(abs(ndc.xy) <= 1.0) && ndc.z >= 0.0 && ndc.z <= 1.0) ? 1.0 : 0.0;
    return o;
}

// The fetches. Nothing here reads across the pixel quad, so the caller may skip it per pixel.
//
// Single return from a pre-initialized local (see ShadowSplitAt): 1.0 is also the right answer for every
// rejected case, since a point this cascade cannot see is a point it knows nothing occluding.
//
// That silence is exactly what makes a mis-sized cascade impossible to diagnose: a receiver outside the
// footprint looks identical to one that nothing occludes, so a shadow that stops at the cascade edge reads
// as a shadow that was never cast. The debug flag inverts the rejected case to "fully occluded", which draws
// the footprint boundary on screen.
//
// Mode ONE, exactly -- not "any debug mode". This used to test `> 0.5`, which meant every mode from 1
// upwards inherited the inversion: mode 2 painted the whole world outside the near cascade yellow, and any
// mode reading the shadow term itself (mode 5 reads the raw coverage) would have been reading a value this
// line had already replaced. A diagnostic that alters what it measures is worse than none.
float ShadowSample(ShadowProjection p, bool isActor, float2 pixel) {
    float lit = abs(shadow_range.y - 1.0) < 0.5 ? 0.0 : 1.0;
    if (p.inside > 0.5) {
        // SOH [Enhancement] Filterable modes replace the whole kernel with one fetch (technique 1). World
        // layer only -- see the note on g_shadowMoments. The mode arrives already reduced to what the
        // backend could actually allocate, so a refused mode reads as 0 here and takes the depth path.
        int filterMode = (int)shadow_jitter.z;
        if (filterMode > 0 && !isActor) {
            float4 stored = g_shadowMoments.SampleLevel(g_shadowMomentSampler, float3(p.uv, p.slice), 0);
            lit = ShadowMomentLit(stored, p.z, filterMode, shadow_jitter.w, shadow_filter.x);
        } else {
            lit = SampleShadowJittered(p.uv, p.z, p.slice, p.texelUv, isActor, pixel);
        }
    }
    return lit;
}

// Dispatch to one cascade with literal indices. The chain covers SHADOW_MAP_MAX_CASCADES entries; if that
// ever grows, this grows with it.
//
// It SELECTS the cascade's constants and then projects once, rather than branching around four separate
// projections. That distinction is the difference between a shader that compiles in milliseconds and one
// that does not: nothing here has a callable form -- ps_4_0 inlines everything -- so a four-arm dispatch
// pasted the whole lookup in four times; ShadowLit calls this twice (the cascade and its cross-fade
// partner) and PSMain calls ShadowLit twice (the world layer and the actor layer), which multiplied out to
// a large number of inlined texture-fetch sites in every receiver shader. FXC at optimisation level 2 takes a long time
// over that, and it runs SYNCHRONOUSLY inside a frame the first time each material is drawn -- which is
// exactly the hitch felt as new geometry rotates into view. Selecting first cuts it to a quarter of that,
// with identical output: only one arm's fetches ever executed anyway.
//
// The split into ShadowProject/ShadowSample does not reopen that: it moves the fetches out of this function
// rather than duplicating them, so the site count is what it was.
//
// Each branch moves four registers' worth of constants, so there is nothing left worth a real branch;
// flattening to conditional moves is cheaper than the jump. Every index stays literal -- see ShadowSplitAt
// for why a computed one cannot be used here.
float ShadowTexelWorldAt(uint cascade) {
    float v = shadow_texel_world.z;
    if (cascade == 0) {
        v = shadow_texel_world.x;
    } else if (cascade == 1) {
        v = shadow_texel_world.y;
    }
    return v;
}

float ShadowActorTexelUvAt(uint cascade) {
    float v = shadow_actor_texel_uv.z;
    if (cascade == 0) {
        v = shadow_actor_texel_uv.x;
    } else if (cascade == 1) {
        v = shadow_actor_texel_uv.y;
    }
    return v;
}

ShadowProjection ShadowProjectAt(float3 p, uint cascade, float sliceBase) {
    float4x4 viewProj = shadow_view_proj[0];
    float texelUv = shadow_texel_uv.x;
    if (cascade == 1) {
        viewProj = shadow_view_proj[1];
        texelUv = shadow_texel_uv.y;
    } else if (cascade == 2) {
        viewProj = shadow_view_proj[2];
        texelUv = shadow_texel_uv.z;
    }
    return ShadowProject(p, viewProj, texelUv, cascade, sliceBase);
}

// First cascade whose far split still covers this depth; the last one catches everything beyond. Zero when
// no cascades were rendered this frame, which is the only case where the answer is not read.
//
// Lifted out of ShadowLitLayers so the diagnostic view can name the cascade a pixel landed in without
// repeating the ladder -- a debug view that reimplements the thing it is inspecting can agree with the
// picture and disagree with the code, which is the one failure mode an instrument may not have. Literal
// indices only, same constraint as ShadowSplitAt.
uint ShadowCascadeIndex(float viewDepth) {
    uint count = (uint)shadow_params.x;
    uint cascade = 0;
    if (count > 0) {
        cascade = count - 1;
        if (viewDepth <= shadow_splits.x) {
            cascade = 0;
        } else if (count > 1 && viewDepth <= shadow_splits.y) {
            cascade = 1;
        } else if (count > 2 && viewDepth <= shadow_splits.z) {
            cascade = 2;
        }
        cascade = min(cascade, count - 1);
    }
    return cascade;
}

// SOH [Enhancement] The cascade's world-to-depth scale, for turning a bias in world units into one in the
// cascade's own normalised depth. The projection is orthographic and row-vector, so the light axis column's
// LENGTH is exactly that factor. Literal indices only, same constraint as ShadowSplitAt.
float ShadowDepthScaleAt(uint cascade) {
    float3 axis = shadow_view_proj[2]._13_23_33;
    if (cascade == 0) {
        axis = shadow_view_proj[0]._13_23_33;
    } else if (cascade == 1) {
        axis = shadow_view_proj[1]._13_23_33;
    }
    return length(axis);
}

// How much the corrections are scaled by this surface's angle to the light (see fast/shadow_map.h).
//
// Acne is a grazing-angle problem: a surface facing the light has none of it, and one edge-on to the light
// has depth running away across a texel. 1/(N.L) is that relationship, and the ceiling is not optional --
// it runs to infinity as the surface turns edge-on, and an unbounded offset there lands the sample in a
// different part of the scene entirely.
float ShadowAcneSlope(float3 normal) {
    if (shadow_acne1.x < 0.5) {
        return 1.0;
    }
    float ndl = saturate(dot(normal, -ShadowLightAxis()));
    return min(1.0 / max(ndl, 1.0e-3), shadow_acne1.y);
}

// Move the sample point off the surface before it is projected.
//
// The normal offset is the one that is correct in principle: the error being corrected is a displacement in
// world space, so the correction is one too -- and being ALONG THE SURFACE rather than along the light, it
// does not detach a shadow from the foot of its caster. The light offset does detach it, which is why it is
// off by default and why it is a separate switch.
float3 ShadowAcneMovePoint(float3 world, float3 normal, float texelWorld, float slope) {
    if (shadow_acne0.x < 0.5) {
        return world;
    }
    float3 moved = world + (normal * (texelWorld * shadow_acne0.y * slope));
    return moved + (-ShadowLightAxis() * (shadow_acne0.z * slope));
}

// Depth bias, in the cascade's normalised depth, from a distance in world units along the light. Subtracted
// from the receiver's depth, so it moves the receiver TOWARDS the light -- the direction that stops a
// surface comparing as occluded by itself.
float ShadowAcneDepthBias(uint cascade, float slope) {
    if (shadow_acne0.x < 0.5) {
        return 0.0;
    }
    return shadow_acne0.w * slope * ShadowDepthScaleAt(cascade);
}

// Pick a cascade by view distance and cross-fade into the next one over the last slice of the range.
// Without the fade the resolution change shows up as a hard line sweeping across the ground as the camera
// moves -- "cascade popping". smoothstep rather than a linear ramp so the seam has no visible corner.
// Returns both caster layers at once: x is the world layer, y the actor layer.
//
// They are one call rather than two because everything except the slice they read is the same for both. The
// cascade choice and the matrix multiply are functions of the receiver, not of which layer is being asked
// about -- so computing them twice was computing them twice identically. Layer L, cascade C is slice
// L*count + C, and that is the whole of the difference: the actor lookup is the world lookup with the layer
// stride added to its slice.
//
// `wantActors` is the receiver kind, constant across a draw call, and it gates only the fetches -- never the
// projection, which has to run for the world layer regardless. So a character pays nothing for the actor
// half it skips, exactly as before, while scenery stops paying twice for the half they share.
float2 ShadowLitLayers(float3 worldPos, float viewDepth, float layerStride, bool wantActors,
                       float2 pixel, float3 normal) {
    // Single return, pre-initialized to "fully lit" -- which is also the answer when no cascades were
    // rendered this frame (count == 0), and for the actor layer whenever this receiver does not take it.
    float2 lit = float2(1.0, 1.0);
    uint count = (uint)shadow_params.x;
    if (count > 0) {
        uint cascade = ShadowCascadeIndex(viewDepth);

        // Past the furthest point any cascade's footprint reaches, there is nothing to look up and the
        // answer is already the one `lit` holds. The bound is computed where the cascades are built, from
        // the boxes themselves rather than from the split ladder -- a cascade's box overshoots its band by a
        // long way, and cutting at the split would take real shadows with it (a low sun throws them well
        // past the band that cast them).
        if (viewDepth <= shadow_range.x) {
            // SOH [Enhancement] Acne corrections (see fast/shadow_map.h). Gated on their own switch here:
            // the ordinary receiver reads the exact surface the depth pass rasterised, so it does not
            // normally need them -- unlike the screen-space mask, which reconstructs its position.
            float3 samplePos = worldPos;
            float acneDepthBias = 0.0;
            if (shadow_acne1.z > 0.5) {
                float slope = ShadowAcneSlope(normal);
                samplePos = ShadowAcneMovePoint(worldPos, normal, ShadowTexelWorldAt(cascade), slope);
                acneDepthBias = ShadowAcneDepthBias(cascade, slope);
            }
            ShadowProjection primary = ShadowProjectAt(samplePos, cascade, 0.0);
            primary.z -= acneDepthBias;

            // Cross-fade band at the far edge of this cascade, where the next one also covers the point.
            // Sampling both and blending is what hides the resolution change; a hard switch draws a visible
            // line that sweeps across the ground as the camera moves.
            bool blend = false;
            float t = 0.0;
            if (cascade + 1 < count) {
                // Not named `far`/`near`: those are legacy Windows macros, and this source is compiled by name
                // at runtime where a stray definition would be baffling to debug.
                float farEdge = ShadowSplitAt(cascade);
                float nearEdge = (cascade == 0) ? 0.0 : ShadowSplitAt(cascade - 1);
                float bandStart = farEdge - (farEdge - nearEdge) * shadow_params.y;
                blend = viewDepth > bandStart;
                t = blend ? smoothstep(bandStart, farEdge, viewDepth) : 0.0;
            }

            // The cross-fade partner, built ONLY where it is read. A matrix select and a multiply, skipped
            // outside the band -- which is a tenth of a cascade's range by default, so this is work that was
            // otherwise thrown away on the large majority of the screen.
            ShadowProjection partner = primary;
            if (blend) {
                uint pc = min(cascade + 1, count - 1);
                partner = ShadowProjectAt(samplePos, pc, 0.0);
                partner.z -= acneDepthBias;
            }

            // Can the actor layer possibly shadow this point at all?
            //
            // That layer holds the characters and nothing else -- a handful of small objects standing somewhere
            // on the map -- yet every scenery pixel on screen was sampling it sixteen fetches deep to be told it
            // was lit. Its world bounding box arrives as a uniform, so a receiver with none of that box behind it
            // along the light can skip the layer's entire kernel. Over a room that is very nearly every scenery
            // pixel, which halves the shadow cost of all of them.
            //
            // The skip is exact, not an approximation. A lookup landing where no actor was drawn reads the
            // cleared depth and returns "lit", which is precisely the value left in place by not doing it. The
            // only thing that has to hold is the direction of the error: the box may be grown, never shrunk.
            bool actorsPossible = false;
            if (wantActors) {
                // The direction the light travels, shared by every cascade (see ShadowLightAxis). Taken here
                // rather than at the top of the block because the slab test below is its only reader, and
                // scenery outside the actor box skips all of it.
                float3 lightAxis = ShadowLightAxis();
                // Everything that can move a lookup off the exact light ray, in world units. The kernel is one
                // bilinear quad, which spans two texels, so it reaches one texel from the sample point; two is
                // that with room to spare. Scaled by the LARGEST cascade's texel, because which cascade this
                // pixel lands in is not decided here for the partner and the unused entries are zero, so the
                // max is both safe and free.
                float texelWorld = max(shadow_texel_world.x, max(shadow_texel_world.y, shadow_texel_world.z));
                float margin = texelWorld * 2.0;
                float3 boxLo = shadow_actor_min.xyz - margin;
                float3 boxHi = shadow_actor_max.xyz + margin;
                // An empty layer arrives inverted and stays inverted once the margin is applied -- the sentinel
                // is many orders of magnitude larger -- so this doubles as the "no characters this frame" test.
                if (all(boxLo <= boxHi)) {
                    // Slab test along the ray from here towards the light. The light TRAVELS along +lightAxis
                    // (see ShadowProject), so anything that can occlude this point lies at -lightAxis from it.
                    //
                    // Zero components are nudged rather than special-cased: a ray parallel to a slab then yields
                    // a huge t of the correct sign, which the min/max below already handle, and never a 0/0.
                    float3 dir = -lightAxis;
                    float3 safeDir = float3(abs(dir.x) < 1e-6 ? 1e-6 : dir.x, abs(dir.y) < 1e-6 ? 1e-6 : dir.y,
                                            abs(dir.z) < 1e-6 ? 1e-6 : dir.z);
                    float3 t0 = (boxLo - worldPos) / safeDir;
                    float3 t1 = (boxHi - worldPos) / safeDir;
                    float3 tNear = min(t0, t1);
                    float3 tFar = max(t0, t1);
                    // Entry is clamped at -margin rather than 0: the point actually projected is already pushed
                    // off the surface and the comparison carries a depth bias, both of which slide the effective
                    // origin a little way towards the light.
                    float tEnter = max(max(tNear.x, tNear.y), max(tNear.z, -margin));
                    float tExit = min(min(tFar.x, tFar.y), tFar.z);
                    actorsPossible = tEnter <= tExit;
                }
            }

            // Four lookups at most -- {this cascade, its cross-fade partner} x {world layer, actor layer} --
            // and they differ in exactly two things: which projection they read and which slice of the array.
            // So they are one loop rather than four pasted copies of a sixteen-tap kernel.
            //
            // [loop] and not [unroll], deliberately, which is the whole point of the shape. Unrolled, the
            // kernel appears four times in the compiled shader; looped, once. This file is compiled by FXC
            // INSIDE a frame, the first time each material is drawn, so its size is not an abstraction -- it is
            // the hitch felt on enabling shadows and on turning the camera into geometry that has not been
            // drawn yet. The same reasoning already cut the four-arm cascade dispatch down to one selection.
            //
            // Nothing here reads across the pixel quad, which is what allows a loop at all: the derivatives are
            // in ShadowProject, above and outside. And the skips mean a pixel executes exactly the lookups it
            // executed before -- a character still does not touch the actor layer, and a pixel outside the band
            // still does not touch the partner.
            [loop]
            for (uint i = 0; i < 4; i++) {
                bool isActor = (i & 1) != 0;
                bool isPartner = i >= 2;
                if (isActor && !wantActors) {
                    continue; // a character is never shadowed by the actor layer, itself included
                }
                if (isActor && !actorsPossible) {
                    continue; // nothing that layer holds is between this point and the light -- see above
                }
                // The actor layer is SHORTER than the world layer -- see SHADOW_MAP_ACTOR_CASCADES. Past its
                // last cascade there is no slice to read, and the answer is the one an empty slice would give.
                // Asked of the cascade index rather than of the projection, so the skip lands before the struct
                // copy below rather than after it.
                if (isActor && (isPartner ? min(cascade + 1, count - 1) : cascade) >= @{o_shadow_actor_cascades}) {
                    continue;
                }
                if (isPartner && !blend) {
                    continue; // outside the band the partner is multiplied by zero, so it is not read
                }
                // Assigned rather than selected with ?:, which HLSL does not offer over a user-defined struct.
                ShadowProjection p = primary;
                if (isPartner) {
                    p = partner;
                }
                // One layer further along the array. Adding the stride to the finished slice is the same number
                // a separate lookup would have built from scratch -- the slice is sliceBase + cascade either
                // way, and both terms are small exact integers.
                if (isActor) {
                    p.slice += layerStride;
                    // The projection is shared with the world layer -- uv and depth are normalised, so they
                    // do not care how big the map is -- but the filter kernel walks in TEXELS, and that layer
                    // may have its own. Equal to the world layer's whenever the two are the same size.
                    p.texelUv = ShadowActorTexelUvAt(isPartner ? min(cascade + 1, count - 1) : cascade);
                }
                float s = ShadowSample(p, isActor, pixel);
                if (isActor) {
                    lit.y = isPartner ? lerp(lit.y, s, t) : s;
                } else {
                    lit.x = isPartner ? lerp(lit.x, s, t) : s;
                }
            }
        } // viewDepth within reach
    }
    return lit;
}
@end

float random(in float3 value) {
    float random = dot(value, float3(12.9898, 78.233, 37.719));
    return frac(sin(random) * 143758.5453);
}

// 3 point texture filtering
// Original author: ArthurCarvalho
// Based on GLSL implementation by twinaphex, mupen64plus-libretro project.

@if(o_three_point_filtering && o_textures[0] || o_textures[1])
cbuffer PerDrawCB : register(b1) {
    struct {
        uint width;
        uint height;
        bool linear_filtering;
    } textures[2];
}

#define TEX_OFFSET(tex, tSampler, texCoord, off, texSize) tex.Sample(tSampler, texCoord - off / texSize)

float4 tex2D3PointFilter(in Texture2D tex, in SamplerState tSampler, in float2 texCoord, in float2 texSize) {
    float2 offset = frac(texCoord * texSize - float2(0.5, 0.5));
    offset -= step(1.0, offset.x + offset.y);
    float4 c0 = TEX_OFFSET(tex, tSampler, texCoord, offset, texSize);
    float4 c1 = TEX_OFFSET(tex, tSampler, texCoord, float2(offset.x - sign(offset.x), offset.y), texSize);
    float4 c2 = TEX_OFFSET(tex, tSampler, texCoord, float2(offset.x, offset.y - sign(offset.y)), texSize);
    return c0 + abs(offset.x)*(c1-c0) + abs(offset.y)*(c2-c0);
}
@end

PSInput VSMain(
    float4 position : POSITION
@for(i in 0..2)
    @if(o_textures[i])
        , float2 uv@{i} : TEXCOORD@{i}
    @end
    @for(j in 0..2)
        @if(o_clamp[i][j])
            @if(j == 0)
                , float texClampS@{i} : TEXCLAMPS@{i}
            @else
                , float texClampT@{i} : TEXCLAMPT@{i}
            @end
        @end
    @end
@end
@if(o_fog)
    , float4 fog : FOG
@end
@if(o_grayscale)
    , float4 grayscale : GRAYSCALE
@end
@if(o_toon || o_shadow_map)
    , float3 normal : NORMAL
@end
@if(o_shadow_map)
    , float4 worldPos : WORLDPOS
@end
@for(i in 0..o_inputs)
    @if(o_alpha)
        , float4 input@{i + 1} : INPUT@{i}
    @else
        , float3 input@{i + 1} : INPUT@{i}
    @end
@end
) {
    PSInput result;
    result.position = position;
    @for(i in 0..2)
        @if(o_textures[i])
            result.uv@{i} = uv@{i};
            @for(j in 0..2)
                @if(o_clamp[i][j])
                    @if(j == 0)
                        result.texClampS@{i} = texClampS@{i};
                    @else
                        result.texClampT@{i} = texClampT@{i};
                    @end
                @end
            @end
        @end
    @end

    @if(o_fog)
        result.fog = fog;
    @end

    @if(o_grayscale)
        result.grayscale = grayscale;
    @end

    @if(o_toon || o_shadow_map)
        result.normal = normal;
    @end

    @if(o_shadow_map)
        result.worldPos = worldPos;
    @end

    @for(i in 0..o_inputs)
        @if(o_alpha)
            result.input@{i + 1} = input@{i + 1};
        @else
            result.input@{i + 1} = float4(input@{i + 1}, 1.0);
        @end
    @end

    return result;
}

@if(o_root_signature)
    [RootSignature(RS)]
@end

@if(srgb_mode)
    float4 fromLinear(float4 linearRGB){
        bool3 cutoff = linearRGB.rgb < float3(0.0031308, 0.0031308, 0.0031308);
        float3 higher = 1.055 * pow(linearRGB.rgb, float3(1.0 / 2.4, 1.0 / 2.4, 1.0 / 2.4)) - float3(0.055, 0.055, 0.055);
        float3 lower = linearRGB.rgb * float3(12.92, 12.92, 12.92);
        return float4(lerp(higher, lower, cutoff), linearRGB.a);
    }
@end

#define MOD(x, y) ((x) - (y) * floor((x)/(y)))
#define WRAP(x, low, high) MOD((x)-(low), (high)-(low)) + (low)

float4 PSMain(PSInput input, float4 screenSpace : SV_Position) : SV_TARGET {
    @for(i in 0..2)
        @if(o_textures[i])
            float2 tc@{i} = input.uv@{i};
            @{s = o_clamp[i][0]}
            @{t = o_clamp[i][1]}
            @if(s || t)
                int2 texSize@{i};
                g_texture@{i}.GetDimensions(texSize@{i}.x, texSize@{i}.y);
                @if(s && t)
                    tc@{i} = clamp(tc@{i}, 0.5 / texSize@{i}, float2(input.texClampS@{i}, input.texClampT@{i}));
                @elseif(s)
                    tc@{i} = float2(clamp(tc@{i}.x, 0.5 / texSize@{i}.x, input.texClampS@{i}), tc@{i}.y);
                @else
                    tc@{i} = float2(tc@{i}.x, clamp(tc@{i}.y, 0.5 / texSize@{i}.y, input.texClampT@{i}));
                @end
            @end

            @if(o_three_point_filtering)
                float4 texVal@{i};
                if (textures[@{i}].linear_filtering) {
                    @if(o_masks[i])
                        texVal@{i} = tex2D3PointFilter(g_texture@{i}, g_sampler@{i}, tc@{i}, float2(textures[@{i}].width, textures[@{i}].height));
                        float2 maskSize@{i};
                        g_textureMask@{i}.GetDimensions(maskSize@{i}.x, maskSize@{i}.y);
                        float4 maskVal@{i} = tex2D3PointFilter(g_textureMask@{i}, g_sampler@{i}, tc@{i}, maskSize@{i});
                        @if(o_blend[i])
                            float4 blendVal@{i} = tex2D3PointFilter(g_textureBlend@{i}, g_sampler@{i}, tc@{i}, float2(textures[@{i}].width, textures[@{i}].height));
                        @else
                            float4 blendVal@{i} = float4(0, 0, 0, 0);
                        @end

                        texVal@{i} = lerp(texVal@{i}, blendVal@{i}, maskVal@{i}.a);
                    @else
                        texVal@{i} = tex2D3PointFilter(g_texture@{i}, g_sampler@{i}, tc@{i}, float2(textures[@{i}].width, textures[@{i}].height));
                    @end
                } else {
                    texVal@{i} = g_texture@{i}.Sample(g_sampler@{i}, tc@{i});
                    @if(o_masks[i])
                        @if(o_blend[i])
                            float4 blendVal@{i} = g_textureBlend@{i}.Sample(g_sampler@{i}, tc@{i});
                        @else
                            float4 blendVal@{i} = float4(0, 0, 0, 0);
                        @end
                        texVal@{i} = lerp(texVal@{i}, blendVal@{i}, g_textureMask@{i}.Sample(g_sampler@{i}, tc@{i}).a);
                    @end
                }
            @else
                float4 texVal@{i} = g_texture@{i}.Sample(g_sampler@{i}, tc@{i});
                @if(o_masks[i])
                    @if(o_blend[i])
                        float4 blendVal@{i} = g_textureBlend@{i}.Sample(g_sampler@{i}, tc@{i});
                    @else
                        float4 blendVal@{i} = float4(0, 0, 0, 0);
                    @end
                    texVal@{i} = lerp(texVal@{i}, blendVal@{i}, g_textureMask@{i}.Sample(g_sampler@{i}, tc@{i}).a);
                @end
            @end
        @end
    @end

    @if(o_alpha) 
        float4 texel;
    @else
        float3 texel;
    @end

    @if(o_2cyc)
        @{f_range = 2}
    @else
        @{f_range = 1}
    @end

    @for(c in 0..f_range)
        @if(c == 1)
            @if(o_alpha)
                @if(o_c[c][1][2] == SHADER_COMBINED)
                    texel.a = WRAP(texel.a, -1.01, 1.01);
                @else
                    texel.a = WRAP(texel.a, -0.51, 1.51);
                @end
            @end

            @if(o_c[c][0][2] == SHADER_COMBINED)
                texel.rgb = WRAP(texel.rgb, -1.01, 1.01);
            @else
                texel.rgb = WRAP(texel.rgb, -0.51, 1.51);
            @end
        @end

        @if(!o_color_alpha_same[c] && o_alpha)
            texel = float4(@{
            append_formula(o_c[c], o_do_single[c][0],
                           o_do_multiply[c][0], o_do_mix[c][0], false, false, true, c == 0)
            }, @{append_formula(o_c[c], o_do_single[c][1],
                           o_do_multiply[c][1], o_do_mix[c][1], true, true, true, c == 0)
            });
        @else
            texel = @{append_formula(o_c[c], o_do_single[c][0],
                           o_do_multiply[c][0], o_do_mix[c][0], o_alpha, false,
                           o_alpha, c == 0)};
        @end
    @end

    @if(o_texture_edge && o_alpha)
        if (texel.a > 0.19) texel.a = 1.0; else discard;
    @end

    texel = WRAP(texel, -0.51, 1.51);
    texel = clamp(texel, 0.0, 1.0);
    // TODO discard if alpha is 0?

    // The alpha cutoff, hoisted to here from the tail of the shader.
    //
    // A discarded fragment writes nothing, so everything computed for it is waste -- and what sits between
    // this point and where the test used to live is the toon relight and the whole shadow lookup, which is
    // the most expensive thing in the file. Cutout geometry is exactly the geometry that discards a large
    // part of every quad it draws, so it was paying full price for the parts of a leaf that are not there.
    // The texture_edge form of the same test has always been above the shadow block; this brings its sibling
    // into line.
    //
    // Hoisting is exact only because nothing in between touches ALPHA: the toon and shadow steps write rgb,
    // fog preserves alpha on the o_alpha path, and grayscale is rgb only. The ONE exception is the noise
    // dither, which multiplies alpha by a per-pixel random mask -- so when that is on, the test has to stay
    // where it was and see the dithered value. That case keeps the original placement below.
    @if(o_alpha && o_alpha_threshold && !o_noise)
        if (texel.a < 8.0 / 256.0) discard;
    @end

    // SOH [Enhancement] Toon lighting: re-light the (white-shaded) albedo with the single dominant
    // light through a soft half-Lambert ramp.
    @if(o_toon)
        float3 toonN = normalize(input.normal);
        float toonNL = dot(toonN, normalize(toon_light_dir)) * 0.5 + 0.5;
        float toonRamp = smoothstep(toon_ramp_center - toon_ramp_softness, toon_ramp_center + toon_ramp_softness, toonNL);
        float3 toonLit = toon_ambient + toon_light_color * toon_highlight_intensity;
        float3 toonShadow = lerp(toonLit, toon_ambient, toon_shadow_intensity);
        if (toon_debug > 0.5) {
            // Diagnostic view: flat white on the lit side of the ramp, flat black in shadow, albedo
            // discarded — makes it obvious which draws are receiving toon lighting.
            texel.rgb = float3(toonRamp, toonRamp, toonRamp);
        } else {
            texel.rgb = clamp(texel.rgb * lerp(toonShadow, toonLit, toonRamp), 0.0, 1.0);
        }
    @end

    // SOH [Enhancement] Cascaded shadow maps: darken where the cascades say this point is occluded.
    // Applied after the toon relight and before fog, so a shadowed surface still fades into the distance
    // like everything else rather than staying dark through the fog.
    @if(o_shadow_map)
        // The vertex normal when the draw has one, and a recovered face normal when it does not. The
        // attribute arrives zeroed on unlit geometry (see the vbo packing), so its length is the test.
        //
        // Nothing on the shipping path reads this any more -- the shadow term is a depth comparison and
        // nothing else, which needs no normal. It survives for diagnostic views 3 and 4, and it is
        // kept because of what those two showed: a recovered face normal is constant across a triangle, so
        // anything downstream that thresholds it prints that triangle's outline, and view 4 is how you find
        // out whether a receiver has a real normal at all. Any future work here will want to ask that
        // question again before assuming an answer.
        //
        // Both computed unconditionally and then selected. ddx/ddy are gradient instructions and may not
        // sit inside varying control flow, so they cannot be moved inside the diagnostic branch that reads
        // them; the cost is two derivatives and a normalize.
        float3 shadowGeoN = cross(ddx(input.worldPos.xyz), ddy(input.worldPos.xyz));
        float shadowNLen = length(input.normal);
        float3 shadowN = (shadowNLen > 1e-4) ? (input.normal / shadowNLen) : normalize(shadowGeoN);
        // input.position.w is the clip-space w the rasterizer interpolated, which for a perspective
        // projection is view depth -- exactly what picks a cascade, with no extra uniform needed.
        // The world caster layer is sampled by everything. The actor layer is sampled only by scenery, so a
        // character is shadowed by the world but never by another character (or by itself) -- the
        // interaction rules the design lays out. Layer L, cascade C is slice L*cascadeCount + C.
        // Scenery also takes the actor caster layer; a character does not, so it is never shadowed by
        // another character or by itself. That choice is constant across a draw call and is passed in, so
        // the character case genuinely skips the second set of taps rather than computing and discarding
        // them -- while the projection the two layers share is built once either way.
        // SOH [Enhancement] The screen-space mask, where it applies (technique 5). One fetch instead of
        // the whole cascade lookup, and a penumbra measured in screen pixels rather than in texels.
        //
        // Scenery only. The mask folds both caster layers together, and a character must never be
        // shadowed by the actor layer -- itself included -- so reading it would paint a character with its
        // own shadow. Characters keep the layered path, which already skips that layer for them.
        //
        // Declared out here because the diagnostic views read the two layers apart (view 2 colours which
        // one occludes), and the mask folds them into a single number. So a view being on also DISABLES
        // the mask below: an instrument that shows something other than what the frame computed is worse
        // than no instrument. shadow_range.y carries the view number; 0 is "shade normally".
        float2 shadowLayers = float2(1.0, 1.0);
        float shadowLit = 1.0;
        bool haveShadow = false;
        if (shadow_mask.x > 0.5 && input.worldPos.w > 0.5 && shadow_range.y < 0.5) {
            float2 maskSample =
                g_shadowMask.SampleLevel(g_shadowMaskSampler, screenSpace.xy * shadow_mask.yz, 0).xy;
            // Same tolerance the mask's own blur used, so a pixel the blur was willing to mix is a pixel
            // the receiver is willing to read. Negative G is the resolve's "nothing was drawn here" marker
            // and fails this by construction.
            float tolerance = shadow_mask.w * max(input.position.w, 1.0) * 0.01;
            if (maskSample.y >= 0.0 && abs(maskSample.y - input.position.w) <= tolerance) {
                shadowLit = maskSample.x;
                haveShadow = true;
            }
        }
        if (!haveShadow) {
            shadowLayers = ShadowLitLayers(input.worldPos.xyz, input.position.w, shadow_params.x,
                                           input.worldPos.w > 0.5, screenSpace.xy, shadowN);
            shadowLit = min(shadowLayers.x, shadowLayers.y);
        }
        // What the comparison produced is COVERAGE -- what fraction of the bilinear quad is occluded -- and
        // it is now shaded with directly. Nothing rewrites it between here and the multiply at the bottom.
        //
        // There used to be a threshold here that remapped coverage through a narrow ramp centred on half,
        // to collapse the gradient into a cel edge, with near and far strengths ramped across the cascade
        // ladder. It is gone with the rest of the mitigations. The edge that is left is the bilinear ramp
        // across one texel of whichever cascade the pixel landed in, which is hard near the camera and
        // softens with distance because the texel does -- unmediated, which is the point.
        //
        // Kept as its own name so debug mode 5 and the shaded picture read the same value. They are now
        // literally the same number, which is worth stating: if the two disagree in SHAPE, something below
        // this line is lying.
        float shadowCoverage = shadowLit;
        // Debug 2: paint the two caster layers apart instead of shading with them. GREEN where the world
        // layer occludes, RED where the actor layer does. A shadow that vanishes is either coming from a
        // layer that stopped capturing or not being sampled at all, and those look identical once the two
        // are combined -- this is the only way to tell which without guessing.
        //
        // DIAGNOSTIC CHANNEL. Modes 1 and 2 show what the shadow system PRODUCED; 3, 4, 5 and 7 show what it
        // was GIVEN. That distinction is the whole reason these exist: a setting can make an artefact less
        // visible and can never say where it came from, because a term that is wrong and a term that is
        // right but badly scaled look identical once they have been multiplied into a colour. These views
        // print the inputs instead, unshaded.
        //
        // Reading them for a faceted or triangular boundary, in the order that narrows fastest:
        //   4 first. If it is red anywhere the artefact appears, the receiver has no vertex normal and a
        //     face normal recovered from screen derivatives is standing in -- constant across a whole
        //     triangle, so anything thresholding it prints that triangle's outline exactly. Dark green is
        //     the subtler version: an interpolated normal sagging towards the 1e-4 test.
        //   3 next, to see it directly. Flat facets of colour on a surface that should be smooth confirm it.
        //   5 to see the shadow term with nothing else multiplied into it. Nothing rewrites coverage any
        //     more, so this and the shaded picture are the same number -- which means a boundary that is
        //     faceted on screen is faceted HERE, in what the depth comparison returned, and the cause is
        //     upstream of everything in this file: the map's own resolution, the matrix it was drawn with,
        //     or the depth pass's bias. Nothing downstream can be blamed for it, because there is nothing
        //     downstream left.
        //   7 to rule out cascade selection entirely -- if the shape follows a cascade boundary it is not a
        //     mesh artefact at all.
        //
        // None of these is a shipping path: each replaces the shaded colour outright. `[branch]` because the
        // mode is a uniform, so a build with the channel off pays one jump and none of the arithmetic.
        uint shadowDebugMode = (uint)(shadow_range.y + 0.5);
        @if(o_fog)
            // Take the fog off the diagnostic. The block below writes a colour that MEANS something, and the
            // fog blend further down would then wash it towards the fog colour in proportion to distance --
            // which is exactly backwards, because distance is where the texels are coarse and the artefact
            // is read. Mode 1's footprint boundary and mode 7's cascade bands sit at the far end by
            // definition and were the worst served. `input` is a by-value copy, so this changes nothing for
            // any other draw.
            if (shadowDebugMode != 0) {
                input.fog.a = 0.0;
            }
        @end
        [branch]
        if (shadowDebugMode == 2) {
            // Which caster layer occludes: GREEN where scenery does, RED where a character does.
            texel.rgb = float3(1.0 - shadowLayers.y, 1.0 - shadowLayers.x, 0.0);
        } else if (shadowDebugMode == 3) {
            // The normal the whole bias chain is built on, as a colour. A smooth surface should shade
            // smoothly; flat patches of one colour are a mesh's triangles showing through.
            texel.rgb = shadowN * 0.5 + 0.5;
        } else if (shadowDebugMode == 4) {
            // Where that normal came from. GREEN = the draw's own vertex normal, and its brightness is the
            // interpolated length, so a triangle whose normal is collapsing towards the 1e-4 test darkens
            // before it flips. RED = no usable normal, so a face normal recovered from ddx/ddy of the world
            // position is being used instead -- constant per triangle by construction.
            texel.rgb = (shadowNLen > 1e-4) ? float3(0.0, saturate(shadowNLen), 0.0) : float3(1.0, 0.0, 0.0);
        } else if (shadowDebugMode == 5) {
            // The coverage the depth comparison produced. Nothing rewrites it before shading now, so this
            // view and the shaded picture carry the same number -- see the reading order above.
            texel.rgb = float3(shadowCoverage, shadowCoverage, shadowCoverage);
        } else if (shadowDebugMode == 7) {
            // Which cascade this pixel sampled: red, green, blue from nearest to furthest. Cross-fade bands
            // read as the primary cascade's colour, since that is the one the picture is keyed to.
            uint shadowDebugCascade = ShadowCascadeIndex(input.position.w);
            texel.rgb = float3(shadowDebugCascade == 0 ? 1.0 : 0.0, shadowDebugCascade == 1 ? 1.0 : 0.0,
                               shadowDebugCascade == 2 ? 1.0 : 0.0);
        } else {
            texel.rgb *= lerp(1.0 - shadow_params.w, 1.0, shadowLit);
        }
    @end

    @if(o_fog)
        @if(o_alpha)
            texel = float4(lerp(texel.rgb, input.fog.rgb, input.fog.a), texel.a);
        @else
            texel = lerp(texel, input.fog.rgb, input.fog.a);
        @end
    @end

    @if(o_grayscale)
        float intensity = (texel.r + texel.g + texel.b) / 3.0;
        float3 new_texel = input.grayscale.rgb * intensity;
        texel.rgb = lerp(texel.rgb, new_texel, input.grayscale.a);
    @end

    @if(o_alpha && o_noise)
        float2 coords = screenSpace.xy * noise_scale;
        texel.a *= round(saturate(random(float3(floor(coords), noise_frame)) + texel.a - 0.5));
    @end

    @if(o_alpha)
        @if(o_alpha_threshold && o_noise)
            // Only reachable when the dither is on; every other case tested this above, before the shading.
            if (texel.a < 8.0 / 256.0) discard;
        @end
        @if(o_invisible)
            texel.a = 0.0;
        @end
        @if(srgb_mode)
            return fromLinear(texel);
        @else
            return texel;
        @end
    @else
        @if(srgb_mode)
            return fromLinear(float4(texel, 1.0));
        @else
            return float4(texel, 1.0);
        @end
    @end
}
