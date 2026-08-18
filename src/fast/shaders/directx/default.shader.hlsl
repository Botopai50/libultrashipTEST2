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
// SamplerComparisonState. Hardware comparison sampling would fold a filtered 2x2 into every fetch for free,
// and it is rejected on merit, not for want of a profile: it can only test all four texels of its footprint
// against ONE depth, and the per-texel depth here is the receiver-plane bias -- the thing that keeps a
// sixteen-tap kernel free of acne without paying for it in peter panning. Gather (see SampleShadowPCF4) buys
// back the fetch count without giving any of that up. The element type is explicit because the fetch has to
// come back as one float, not a float4 to be truncated.
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

// Everything is float4-shaped on purpose: HLSL gives each element of a `float arr[n]` its own 16-byte
// register, so a scalar array would waste three quarters of its space and make the C++ layout easy to get
// subtly wrong. Layout matches the PerShadowCB C++ struct exactly.
cbuffer PerShadowCB : register(b3) {
    row_major float4x4 shadow_view_proj[@{o_shadow_max_cascades}];
    float4 shadow_splits;      // far distance of each cascade, world units
    float4 shadow_texel_world; // world size of one texel, per cascade
    float4 shadow_texel_uv;    // one texel in UV terms (1/resolution), per cascade
    // Constant bias in NDC depth, per cascade. Held per cascade rather than as one number because each
    // cascade covers a different depth range: a single NDC value would mean a different physical distance
    // in each one, which detaches distant shadows and makes the same shadow land in two places across a
    // cascade transition.
    float4 shadow_depth_bias;
    // x = active cascade count (0 = no shadow map this frame), y = cross-fade band as a fraction of the
    // cascade, z = receiver push along the normal in texels, w = darkness where fully occluded.
    float4 shadow_params;
    // x = PCF kernel radius in texels (see SHADOW_MAP_DEFAULT_FILTER_WIDTH)
    // y = debug view selector. Modes 1 and 2 show what the system produced; 3 to 8 show what it was given,
    //     which is what an artefact of unknown origin needs -- see the channel in PSMain for how to read
    //     them against each other.
    //     0 off
    //     1 everything OUTSIDE a cascade's footprint painted fully occluded instead of silently lit, which
    //       is the only way to see where a cascade actually ends
    //     2 the two caster layers by colour: green = occluded by the world layer, red = by the actor layer
    //     3 the normal the bias chain runs on, as a colour
    //     4 where that normal came from: green = the draw's vertex normal (brightness = its interpolated
    //       length), red = a face normal recovered from screen derivatives
    //     5 the filter's raw coverage, before the hardening remap rewrites it
    //     6 receiver-plane gradient magnitude, red where the clamp bound
    //     7 cascade index, red/green/blue from nearest to furthest
    //     8 edge hardness after the incidence taper, red = hard, green = soft
    //     9 the PCF kernel's effective reach in texels, red = zero (the filter is doing nothing)
    // z = edge hardness near, w = edge hardness in the furthest cascade, ramped between across the ladder
    //     (see SHADOW_MAP_DEFAULT_EDGE_HARDNESS).
    float4 shadow_filter;
    // The incidence band, which now shapes the edge HARDENING only. It used to gate whether the shadow was
    // applied at all; that fell to the tuned configuration, since a wall keeping its shadow is worth more
    // than a boundary that is slightly rough. What remains is a taper: the threshold is only as trustworthy
    // as the contour it traces, and that contour coarsens as a surface turns edge-on to the light, so the
    // hardening eases off over this band rather than drawing a hard line through a coarse one.
    //   x = incidence below which the hardening is at its floor
    //   y = incidence at which the hardening is at full strength
    //   z = that floor, as a fraction of the configured hardness
    //   w = NOT an incidence value: the furthest view depth any cascade's footprint reaches, riding in the
    //     spare slot rather than growing the buffer for one float. Past it there is nothing to look up.
    // In the constant buffer rather than as template symbols so they can be dialled while the game runs --
    // a compile-time symbol makes every trial a rebuild.
    float4 shadow_incidence;
    // World-space bounds of the ACTOR caster layer. Everything in that layer is inside this box, so a
    // receiver with no part of the box behind it along the light cannot be shadowed by it -- see the test in
    // ShadowLitLayers. An empty layer arrives inverted and fails every test, which is what "no characters"
    // should do.
    float4 shadow_actor_min;
    float4 shadow_actor_max;
    // One texel of the ACTOR layer in UV terms, per cascade. Equal to shadow_texel_uv while the two layers
    // share a resolution; separate once that layer is sized on its own (see fast/shadow_map.h).
    float4 shadow_actor_texel_uv;
    // The receiver-plane gradient's bound and what happens at it.
    //   x = the bound itself, in the cascade's normalised units (see SHADOW_MAP_DEFAULT_PLANE_GRADIENT_LIMIT)
    //   y = 1 to also narrow the kernel by the overshoot, 0 to truncate the gradient and nothing else
    //   z = most bilinear quads the kernel may lay along the receding direction; 1 disables the anisotropic
    //       path entirely and every pixel takes the square kernel it always took
    //   w = 1 to measure the hard edge's ramp in screen pixels rather than in coverage
    // In the constant buffer rather than as literals so the bound can be swept while the game runs: it is
    // the last suspect standing for the faceted banding, and a suspect that needs a rebuild per trial is a
    // suspect that never gets tested.
    float4 shadow_plane;
    // x = spacing between the anisotropic kernel's quads, in texels. y, z, w reserved.
    //
    // Its own register rather than riding in a spare component of something else: it is not a plane-bias
    // value and has nothing to do with the four beside it, and a number parked where it does not belong is
    // how a constant buffer becomes unreadable.
    float4 shadow_aniso;
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
float SampleShadowPCF4(float2 uv, float2 uvCentre, float2 grad, float2 gradTexel, float z, float slice,
                       float texelUv, bool isActor) {
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
    // That last part is why this is NOT hardware comparison sampling. A comparison sampler would also fold
    // the four compares and the blend into the fetch, but it can only test all four texels against ONE
    // depth -- and the per-texel depth is the receiver-plane bias, which is the thing keeping the wide
    // kernel free of acne without paying for it in peter panning. Gather gives up none of it.
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

    // The four depths the receiver's own plane has at those texels.
    //
    // uvCentre, not uv: every tap in the whole kernel measures its plane correction from the ONE point the
    // receiver's depth was evaluated at, otherwise each 2x2 quad would correct against itself and the
    // sixteen-tap kernel would still disagree with itself across its own width.
    //
    // One dot for the corner and three adds for the rest, rather than a dot per texel. Stepping one texel
    // along an axis moves the plane by gradTexel on that axis -- which the caller already has, since it is
    // the same for all four quads -- so the other three corners are the first one plus a constant.
    //
    // Same value, reassociated: `base + t*grad.x` rounds where `dot(offset + (t,0), grad)` did not, so the
    // two can land on opposite sides of a comparison that is an exact tie. Measured over four million
    // adversarial samples -- stored depths deliberately clustered onto the threshold -- they disagreed on
    // 0.001% of them, and every one of those was within ONE ulp of a tie. The visible consequence of such a
    // disagreement is one tap in sixteen, on a pixel whose surface depth equals the stored depth to the last
    // representable bit.
    float base = z + dot(uv00 - uvCentre, grad);
    float4 refs = base + float4(gradTexel.y, gradTexel.x + gradTexel.y, gradTexel.x, 0.0);

    // step(a, b) is b >= a, so this is "the plane is at or in front of the stored depth" -- 1 where the
    // texel does not occlude -- for all four at once.
    float4 lit = step(refs, stored);

    // Bilinear weights, written out. lerp(lerp(w, z, sx), lerp(x, y, sx), sy) is exactly this sum, and as a
    // dot it is one instruction instead of three dependent ones. Comparing first and filtering after is the
    // whole point of the kernel -- filtering the stored depths and comparing once would give a wrong
    // penumbra -- and that ordering is unchanged: `lit` is already the comparison result.
    float2 inv = 1.0 - subTexel;
    float4 weights = float4(inv.x * subTexel.y, subTexel.x * subTexel.y, subTexel.x * inv.y, inv.x * inv.y);
    return dot(lit, weights);
}

// Wider filter: four bilinear taps one texel apart, covering a 4x4 texel neighbourhood. Sixteen fetches
// instead of four.
//
// This is the only lever on distant jaggedness that costs neither range nor memory. Texel size is what
// quantizes a shadow edge, and the far cascade spends 2048 texels on a radius of about 4000 world units --
// four units per texel. Shortening its range or raising the resolution both fix that directly but cost
// something; a wider filter cannot make the edge more accurate, it can only spread the step over enough
// pixels to stop reading as a staircase. On this kind of workload the extra fetches are the cheapest of
// the three currencies.
//
// Redistributing the cascade splits was checked first and does not help: the far cascade's radius comes
// mostly from the frustum's lateral spread at its far edge, not from how long the slice is, so moving the
// split only trades the near cascades (already ~36x oversampled) for almost nothing.
float SampleShadowPCF16(float2 uv, float2 grad, float z, float slice, float texelUv, bool isActor,
                        float kernelScale, float anisoStretch) {
    // Spacing is a tunable radius in texels, NOT a free parameter: each bilinear tap already spans a 2x2
    // texel quad, so a radius of one texel puts those quads edge to edge and covers 4x4 contiguously, and
    // anything WIDER leaves texels between the quads sampled by nothing -- a regular hole in the kernel,
    // which reads on screen as a grid laid over the ground. Two texels did exactly that. Below one the
    // quads overlap instead, which only costs redundancy, so this is safe to turn down for a tighter
    // penumbra and must not be turned above 1.0.
    float d = texelUv * min(shadow_filter.x, 1.0) * kernelScale;
    // How far the receiver plane's depth moves across one texel, per axis. Identical for all four quads --
    // they share a cascade, so they share its texel -- so it is formed once here rather than four times
    // inside them.
    float2 gradTexel = grad * texelUv;

    // The square kernel, unchanged, and the only one built when the anisotropic path is off. `[branch]` on a
    // value that is uniform across a draw whenever the feature is off, so a build that does not use it pays
    // one jump and none of the rest.
    [branch]
    if (anisoStretch <= 0.0) {
        float sum = SampleShadowPCF4(uv + float2(-d, -d), uv, grad, gradTexel, z, slice, texelUv, isActor);
        sum += SampleShadowPCF4(uv + float2(d, -d), uv, grad, gradTexel, z, slice, texelUv, isActor);
        sum += SampleShadowPCF4(uv + float2(-d, d), uv, grad, gradTexel, z, slice, texelUv, isActor);
        sum += SampleShadowPCF4(uv + float2(d, d), uv, grad, gradTexel, z, slice, texelUv, isActor);
        return sum * 0.25;
    }

    // Stretched along the direction the receiver recedes from the light, which is the direction the gradient
    // already points in: grad is depth per unit uv, so it is steepest exactly where the surface runs away
    // fastest, and that is the axis the map's texel steps are magnified along. Normalising it costs one
    // rsqrt and needs no extra uniform.
    //
    // A row of quads along that axis, two texels apart so they tile it with no gap, and the row is two quads
    // deep across it -- the perpendicular direction is sampled perfectly well and gets no widening at all.
    // That asymmetry is the whole point: widening in both directions would smear the edges that are correct
    // in order to fix the one that is not.
    //
    // `[loop]` and not `[unroll]`, for the same reason the layer lookup below is a loop: unrolled, the whole
    // quad appears once per iteration in the compiled shader, and this file is compiled by FXC inside a
    // frame the first time each material draws.
    float mag = length(grad);
    float2 axis = mag > 1e-6 ? (grad / mag) : float2(1.0, 0.0);
    float2 perp = float2(-axis.y, axis.x);
    float2 across = perp * d;
    // The count is UNIFORM -- read from the setting, not from this pixel's gradient. Deriving it per pixel
    // is what put a per-triangle step into the filter; the stretch is carried by the spacing instead, which
    // is a float and varies smoothly.
    uint taps = (uint)max(shadow_plane.z, 1.0);
    // Spaced so the row spans the stretch, never wider than the no-gap setting. Below that the row simply
    // draws in: at a stretch of one it is about a texel across, which is the square kernel again, reached
    // continuously rather than by switching.
    //
    // kernelScale multiplies this, and leaving it out was a defect with a visible signature. That factor is
    // the plane bound's taper: where the receiver's gradient overshoots the bound, the depth correction each
    // tap carries is truncated, and the residue every tap suffers is its DISTANCE from the sample point
    // times how far the gradient overshot. The taper exists to hold that product still by pulling the taps
    // in -- and it was pulling in only the two across the row, never the row itself.
    //
    // Which is backwards, because the row lies along the direction depth runs away fastest. So on exactly
    // the grazing surfaces where the correction stops being exact, the kernel was reaching furthest along
    // the worst axis with a truncated correction, and the residue came out as regular banding at the row's
    // own period. Striping with a period is a sampling structure, not geometry, and this is the structure.
    float anisoSpacing = min(max(shadow_aniso.x, 0.05), anisoStretch / (float)taps) * kernelScale;
    // Weighted, not averaged, and the difference is the whole look.
    //
    // A flat average is a BOX filter, and a box has an abrupt end. Convolved with a shadow boundary it does
    // produce a ramp, but the ramp's slope jumps where the box's end crosses the boundary -- so as the
    // surface varies by a texel the half-coverage contour does not slide, it HOPS. Threshold that for a cel
    // edge and every hop is a kink: the line comes out scribbled rather than drawn, which is exactly the
    // artefact this weighting exists to remove. Widening a box does not help, because the discontinuity is
    // at its ends however far apart they are.
    //
    // The weight below goes to zero smoothly at both ends and has zero slope there too, so no tap enters or
    // leaves the sum abruptly and the contour moves continuously with the surface. It costs nothing: same
    // taps, same fetches, three arithmetic operations per column.
    //
    // What it does cost is nominal reach. The outer taps count for little, so the EFFECTIVE span is around
    // three fifths of taps * spacing -- which is why the defaults buy more span than the step strictly
    // needs, and why raising the tap count is the way to answer a step this does not cover.
    float sum = 0.0;
    float weightSum = 0.0;
    const float halfSpan = max(((float)taps - 1.0) * 0.5, 1e-6);
    [loop]
    for (uint i = 0; i < taps; i++) {
        // Centred on the sample point, so the kernel stays symmetric about the pixel it is shading rather
        // than reaching only one way along the light.
        float step01 = (float)i - ((float)taps - 1.0) * 0.5;
        float offsetAlong = step01 * texelUv * anisoSpacing;
        float2 along = axis * offsetAlong;
        // Normalised position across the row, then the smooth compact weight. Squared so the falloff is
        // flat at the ends rather than merely zero there.
        float tNorm = step01 / halfSpan;
        float wBase = saturate(1.0 - (tNorm * tNorm));
        float w = wBase * wBase;
        weightSum += 2.0 * w;
        sum += w * SampleShadowPCF4(uv + along + across, uv, grad, gradTexel, z, slice, texelUv, isActor);
        sum += w * SampleShadowPCF4(uv + along - across, uv, grad, gradTexel, z, slice, texelUv, isActor);
    }
    return sum / max(weightSum, 1e-6);
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
// One cascade lookup, split into the half that must run in uniform control flow and the half that must not.
//
// The projection below ends in screen-space derivatives, and those are gradient instructions: they read
// across the pixel quad, so they are only defined where every pixel of the quad executes them. The sixteen
// texture fetches that follow carry no such requirement -- SampleLevel takes an explicit LOD -- so once they
// are separated they can sit behind a real branch.
//
// Welded together, they could not. The cross-fade partner's entire kernel had to be evaluated for every
// pixel and then multiplied by a blend weight that is zero outside a band covering a tenth of a cascade's
// range: sixteen fetches per layer, thrown away on the large majority of the screen. Splitting is what lets
// ShadowLit skip them instead, and it also removes a latent hazard -- derivatives taken inside flow control
// that is not uniform have undefined results, which is what the old shape asked for.
struct ShadowProjection {
    float2 uv;
    float2 grad;
    float z;      // ndc depth with the constant bias already subtracted
    float texelUv;
    float slice;  // texture-array slice, this layer's offset included
    float inside; // 1 where the cascade covers this point, 0 where there is nothing to sample
    // Multiplies the PCF kernel's reach. 1 everywhere unless the plane gradient overshot its bound and the
    // soft falloff is on, in which case it is limit/gradient -- see ShadowPlaneGradient.
    float kernelScale;
    // How many bilinear quads the kernel lays along the receding direction. 1 is the square kernel, which is
    // what every pixel gets while the anisotropic path is off.
    float anisoStretch;
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

// How far, and along what, to push the sample off the surface before projecting it. Cascade-independent for
// the same reason the axis above is, so it is also computed once per pixel and passed in; only the SIZE of
// the push varies per cascade, and that is one multiply by texelWorld at the point of use.
//
// A depth-only bias cannot fix curved surfaces -- it only slides the comparison along the light ray, still
// inside the same polygon -- whereas this moves it sideways, out of the geometry casting onto itself. That
// is what removes the striped self-shadowing (acne). normalWs always arrives unit length -- the caller
// resolves the vertex normal or a recovered face normal before this point -- so the term is always live.
//
// Orient the normal to face the light before pushing along it. The offset only helps if it moves the sample
// OFF the surface towards the light; pushed the other way it drives the sample into the geometry and makes
// the self-shadowing worse than no offset at all. That sign is not something the caller can guarantee: a
// normal recovered from screen-space derivatives comes out either way depending on triangle winding and
// which way the screen's y axis runs, and a vertex normal can disagree with the face it sits on. A surface
// facing the light has a normal pointing against the direction the light travels.
//
// Scale the push by how obliquely the light strikes this surface, sin of the angle between them.
//
// Acne is a grazing-angle artefact: depth changes fast across a texel exactly when the surface is nearly
// edge-on to the light, and hardly at all when it faces the light square. Applying the same push everywhere
// therefore spent its whole cost where it bought nothing -- on a floor lit from overhead the offset is pure
// displacement along the light ray, which is peter panning and nothing else, and it is precisely on such
// floors that a shadow's contact point is most closely read.
//
// sin also has the right shape at the other end: it goes to 1 as the surface turns edge-on, which is where
// the offset stops displacing along the ray at all and starts sliding sideways off the polygon, which is the
// only thing that actually removes the stripes. So this is not a trade of one artefact for the other -- it
// moves the whole budget to the angles that need it.
float3 ShadowNormalOffset(float3 normalWs, float3 lightAxis) {
    float3 n = dot(normalWs, lightAxis) > 0.0 ? -normalWs : normalWs;
    float ndotl = saturate(dot(n, -lightAxis));
    float grazing = sqrt(saturate(1.0 - (ndotl * ndotl)));
    return n * (shadow_params.z * grazing);
}

// The receiver plane expressed in shadow-map coordinates: how fast its depth changes per unit of uv. The
// two derivative pairs give depth and uv per screen pixel, and inverting the uv Jacobian turns that into
// depth per uv. SampleShadowPCF4 uses it to compare each texel against the plane rather than against one
// point on it.
//
// SEPARATE from the projection, and called only from flow every pixel reaches, because it is the only part
// of this that may not be branched around: screen-space derivatives taken where neighbouring pixels of a
// quad took different paths are undefined.
//
// It is also the same answer for EVERY cascade, which is what lets one call serve both projections. The
// cascade's radius scales uv and its depth range scales z, and the depth range is five times the radius by
// construction -- so the radius cancels and the gradient is a property of the surface and the light, not of
// the cascade looking at it. Checked numerically against pairs of cascades with independent radii and
// centres: the two agree to 2e-10 relative, which is the arithmetic's own noise.
// Returns the bounded gradient in xy, the kernel scale the bound leaves behind in z, and in w how many
// bilinear quads the kernel should lay along the direction the receiver recedes -- see the notes at the
// bottom of the body.
float4 ShadowPlaneGradient(float3 p, float3 lx, float3 ly, float3 lz) {
    float3 dpx = ddx(p);
    float3 dpy = ddy(p);
    // The uv and depth derivatives, each missing its cascade's own scale factor: uv carries 0.5/radius and
    // depth carries 1/(5*radius). Both are dropped here and put back as the single 2/5 below, which is what
    // the two scales reduce to once the solve divides one by the other -- see the constant.
    float2 duvdx = float2(dot(dpx, lx), -dot(dpx, ly));
    float2 duvdy = float2(dot(dpy, lx), -dot(dpy, ly));
    float dzdx = dot(dpx, lz);
    float dzdy = dot(dpy, lz);
    float det = (duvdx.x * duvdy.y) - (duvdx.y * duvdy.x);
    float2 grad = float2(0.0, 0.0);
    if (abs(det) > 1e-12) {
        // 0.4 is the 2/5 the radius collapses to. A cascade's depth range is five times its radius by
        // construction, so the radius in the uv scale and the radius in the depth scale cancel and leave a
        // pure number -- which is the whole reason this can be computed without knowing which cascade will
        // read it. Verified against the projection-based form it replaces over 300000 random surfaces and
        // cascades: 4.6e-10 relative, the arithmetic's own noise.
        grad = float2((duvdy.y * dzdx) - (duvdx.y * dzdy), (duvdx.x * dzdy) - (duvdy.x * dzdx)) * (0.4 / det);
    }
    // Bound it. At a silhouette the quad straddles two surfaces and the derivative is meaningless, and an
    // unbounded correction there would punch a hole through the shadow. A surface at forty-five degrees to
    // the light has a gradient of 2/5; the default bound is eight times that, about eighty-three degrees,
    // past which a receiver is edge-on enough that the constant and normal-offset terms are the right tools.
    //
    // Returned with a KERNEL SCALE alongside it, because truncating the gradient does not make the
    // correction safe -- it makes it wrong in a particular direction. Past the bound the taps compare
    // against a plane flatter than the surface they sit on, and that error is the product of the overshoot
    // and how far out the tap reaches. Nothing here can fix the overshoot, so the scale takes the other
    // factor: shrink the kernel by exactly limit/gradient and the product holds at the value it had AT the
    // bound. Continuous in the gradient, which is the point -- a threshold on a per-face quantity is what
    // prints a mesh's faces onto the screen.
    //
    // Off (shadow_plane.y == 0) the scale is a constant 1 and this is the truncation it always was.
    float lim = shadow_plane.x;
    float mag = max(abs(grad.x), abs(grad.y));
    float kernelScale = 1.0;
    if (shadow_plane.y > 0.5 && mag > lim) {
        kernelScale = lim / mag;
    }

    // How many quads the kernel needs along the direction this receiver recedes from the light.
    //
    // This is the projective-aliasing term, and it is a different animal from everything else in this file.
    // Acne is a false comparison and a bias fixes it. This is a SAMPLING limit: on a surface turning edge-on
    // to the light the map has almost no resolution along the direction the surface recedes, so the shadow
    // boundary quantises into steps of one texel over the sine of the angle. At the bound above -- about
    // eighty-three degrees -- that is eight texels per step, and at eighty-seven it is twenty. No bias moves
    // them, because nothing is mis-compared: the boundary is being drawn at a resolution that does not exist.
    // Those steps, crossed at an angle by a shadow's edge, are the teeth.
    //
    // What CAN be done is average over a whole step instead of resolving it, which turns a hard staircase
    // into a soft directional gradient -- the same information, presented as a penumbra rather than as
    // saw-teeth. That needs the kernel widened along the receding direction and NOT across it, where the map
    // samples perfectly well and a wider kernel would only smear a good edge.
    //
    // The magnitude is read straight off the gradient, which already measures exactly this: |grad| is
    // (2/5)cot(alpha), so |grad|/0.4 is cot(alpha), which for a grazing surface is the step magnification
    // 1/sin(alpha) to within a percent or two. Taken from the RAW magnitude, before the bound above, because
    // the bound is about how far the depth correction may be trusted and this is about how far the sampling
    // is stretched -- two different questions that happened to share a number.
    //
    // Widening is spent on MORE quads rather than on wider spacing, which is not a detail: each bilinear
    // quad spans two texels, so quads set two texels apart tile the run contiguously, and anything further
    // leaves texels between them sampled by nothing. That is a regular hole in the kernel, and it reads on
    // screen as a grid -- trading teeth for stripes. The count is what widens; the spacing never does.
    // How far the sampling is stretched along the receding direction, as a continuous multiple.
    //
    // This used to round to a TAP COUNT here, and rounding was a mistake of exactly the kind this whole
    // investigation was about. |grad| comes from screen derivatives of a planar function, so it is constant
    // across a triangle; rounding it is a threshold on a per-triangle constant, and a threshold on a
    // per-triangle constant prints that triangle's outline. Two neighbouring faces landed on four taps and
    // five, got rows of different length, blurred by different amounts, and the seam between them was the
    // edge they share. The artefact the kernel exists to remove, reintroduced by the kernel.
    //
    // Continuous now, and the tap COUNT is uniform -- see SampleShadowPCF16, which spaces a fixed number of
    // quads to span this. A clamp still bounds it, but a clamp leaves the value continuous where a round
    // does not: only the slope kinks, and a kink does not draw an outline.
    //
    // Zero means the anisotropic path is off, which is uniform across the draw.
    float anisoStretch = 0.0;
    if (shadow_plane.z > 1.0) {
        // Bounded by the REACH the row can actually cover, which is its tap count times its spacing -- not
        // by the tap count alone. Capping at the count was an arithmetic mistake with a decisive
        // consequence: spacing is min(setting, stretch/taps), so a stretch that could never exceed the count
        // forced the spacing to one texel or less and the row to `taps` texels of reach, whatever either
        // setting said.
        //
        // That is why widening the row never did anything. The step it has to average over is one texel
        // over the sine of the angle -- around fourteen texels on a wall this grazing -- and five taps
        // capped this way reach five. No value of any control could close that, because the cap was
        // upstream of all of them.
        float reach = shadow_plane.z * max(shadow_aniso.x, 0.05);
        anisoStretch = clamp(length(grad) / 0.4, 1.0, max(reach, 1.0));
    }
    return float4(clamp(grad, -lim, lim), kernelScale, anisoStretch);
}

// Everything about a lookup EXCEPT the gradient, which the caller fills in. No derivatives here, which is
// what makes this safe to call from inside a branch -- and the partner projection is now built only where
// it is actually read.
//
// `ndcZ` comes back separately because the gradient needs the depth as projected, while the struct carries
// it with the cascade's constant bias already taken off.
ShadowProjection ShadowProject(float3 p, float4x4 viewProj, float texelUv, float depthBias, uint cascade,
                               float sliceBase, float4 gradScale) {
    float4 clip = mul(float4(p, 1.0), viewProj);

    float safeW = abs(clip.w) > 1e-6 ? clip.w : 1e-6;
    float3 ndc = clip.xyz / safeW;
    // NDC -> texture space (y flips: NDC is +up, textures are +down).
    float2 uv = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    ShadowProjection o;
    o.uv = uv;
    o.grad = gradScale.xy;
    o.kernelScale = gradScale.z;
    o.anisoStretch = gradScale.w;
    o.z = ndc.z - depthBias;
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
float ShadowSample(ShadowProjection p, bool isActor) {
    float lit = abs(shadow_filter.y - 1.0) < 0.5 ? 0.0 : 1.0;
    if (p.inside > 0.5) {
        lit = SampleShadowPCF16(p.uv, p.grad, p.z, p.slice, p.texelUv, isActor, p.kernelScale, p.anisoStretch);
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
// 256 inlined texture-fetch sites in every receiver shader. FXC at optimisation level 2 takes a long time
// over that, and it runs SYNCHRONOUSLY inside a frame the first time each material is drawn -- which is
// exactly the hitch felt as new geometry rotates into view. Selecting first cuts it to 64 sites with
// identical output: only one arm's fetches ever executed anyway.
//
// The split into ShadowProject/ShadowSample does not reopen that: it moves the fetches out of this function
// rather than duplicating them, so the site count is what it was.
//
// Each branch moves four registers' worth of constants, so there is nothing left worth a real branch;
// flattening to conditional moves is cheaper than the jump. Every index stays literal -- see ShadowSplitAt
// for why a computed one cannot be used here.
// The world size of one texel, on its own, because the point that gets projected depends on it and the
// gradient is taken from that point -- which now happens before any of the rest is selected.
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

ShadowProjection ShadowProjectAt(float3 p, uint cascade, float sliceBase, float4 gradScale) {
    float4x4 viewProj = shadow_view_proj[0];
    float texelUv = shadow_texel_uv.x;
    float depthBias = shadow_depth_bias.x;
    if (cascade == 1) {
        viewProj = shadow_view_proj[1];
        texelUv = shadow_texel_uv.y;
        depthBias = shadow_depth_bias.y;
    } else if (cascade == 2) {
        viewProj = shadow_view_proj[2];
        texelUv = shadow_texel_uv.z;
        depthBias = shadow_depth_bias.z;
    }
    return ShadowProject(p, viewProj, texelUv, depthBias, cascade, sliceBase, gradScale);
}

// Which band of the cascade ladder this depth falls in, normalised 0 (nearest) to 1 (furthest).
//
// Used to ramp the edge hardness with distance, because the artefact it fights is not one size: the kernel
// is three texels wide in every cascade, but a texel of the near one is a fraction of a world unit and one
// of the far one is several, so a single hardness leaves the near edge crisp and the far edge metres wide.
// Literal indices only, same constraint as ShadowSplitAt, and a single return over a pre-initialised local.
float ShadowLadderFraction(float viewDepth) {
    float band = 0.0;
    uint count = (uint)shadow_params.x;
    if (count > 1) {
        float step = 0.0;
        if (viewDepth > shadow_splits.x) {
            step = 1.0;
        }
        if (viewDepth > shadow_splits.y) {
            step = 2.0;
        }
        band = saturate(step / (float)(count - 1));
    }
    return band;
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

// Pick a cascade by view distance and cross-fade into the next one over the last slice of the range.
// Without the fade the resolution change shows up as a hard line sweeping across the ground as the camera
// moves -- "cascade popping". smoothstep rather than a linear ramp so the seam has no visible corner.
// Returns both caster layers at once: x is the world layer, y the actor layer.
//
// They are one call rather than two because everything except the slice they read is the same for both. The
// cascade choice, the normal-offset push, the matrix multiply, the screen-space derivatives and the receiver
// plane gradient are all functions of the receiver, not of which layer is being asked about -- so computing
// them twice was computing them twice identically. Layer L, cascade C is slice L*count + C, and that is the
// whole of the difference: the actor lookup is the world lookup with the layer stride added to its slice.
//
// `wantActors` is the receiver kind, constant across a draw call, and it gates only the fetches -- never the
// projection, which has to run for the world layer regardless. So a character pays nothing for the actor
// half it skips, exactly as before, while scenery stops paying twice for the half they share.
//
// `gradOut` hands the receiver-plane gradient back for the diagnostic views. It is not an extra
// computation: the gradient is on the critical path of every comparison below whether anything reads it or
// not, so this is a register move and nothing else. Debug mode 6 paints it, which is the only way to see
// the clamp inside ShadowPlaneGradient binding -- and a clamp that binds on some triangles of a mesh and
// not on others is one of the ways a triangular artefact gets drawn.
float2 ShadowLitLayers(float3 worldPos, float3 normalWs, float viewDepth, float layerStride, bool wantActors,
                       out float3 gradOut) {
    // Single return, pre-initialized to "fully lit" -- which is also the answer when no cascades were
    // rendered this frame (count == 0), and for the actor layer whenever this receiver does not take it.
    // gradOut is pre-initialized for the same reason, and because an out parameter the compiler cannot
    // prove is assigned on every path draws the "potentially uninitialized" warning ShadowSplitAt documents.
    float2 lit = float2(1.0, 1.0);
    // z is the kernel scale, which is 1 wherever nothing has narrowed the filter -- the same value a pixel
    // that never reaches a cascade would have used.
    gradOut = float3(0.0, 0.0, 1.0);
    uint count = (uint)shadow_params.x;
    if (count > 0) {
        uint cascade = ShadowCascadeIndex(viewDepth);

        // The light's basis, and everything the two projections agree on. Each cascade's matrix holds these
        // three axes scaled by its own factors, so normalising the columns of any of them gives the same
        // vectors -- cascade 0's are taken. The push along the normal follows: same direction for every
        // cascade, only its length differs.
        float3 lightAxis = ShadowLightAxis();
        float3 lightX = normalize(shadow_view_proj[0]._11_21_31);
        float3 lightY = normalize(shadow_view_proj[0]._12_22_32);
        float3 offsetDir = ShadowNormalOffset(normalWs, lightAxis);

        // The point that gets projected, and the gradient of the receiver plane through it. This much runs
        // for every pixel and has to: the gradient is where the screen-space derivatives are taken, and
        // those are only defined in flow every pixel of a quad reaches.
        //
        // It is also ALL that has to. The gradient used to be read off a finished projection, which meant a
        // matrix select and a multiply had to happen first, for every pixel, before anything could be
        // decided. Taken from the point's light-space derivatives instead it needs no projection at all --
        // and everything downstream becomes branchable, which is what the two skips below are made of.
        float3 sample = worldPos + offsetDir * ShadowTexelWorldAt(cascade);
        // xy is the bounded gradient, z the kernel scale its bound left behind (1 unless the soft falloff
        // is on and the gradient overshot). Both travel together into every projection this pixel makes.
        float4 grad = ShadowPlaneGradient(sample, lightX, lightY, lightAxis);
        gradOut = grad.xyz;

        // Past the furthest point any cascade's footprint reaches, there is nothing to look up and the
        // answer is already the one `lit` holds. The bound is computed where the cascades are built, from
        // the boxes themselves rather than from the split ladder -- a cascade's box overshoots its band by a
        // long way, and cutting at the split would take real shadows with it (a low sun throws them well
        // past the band that cast them).
        if (viewDepth <= shadow_incidence.w) {
            ShadowProjection primary = ShadowProjectAt(sample, cascade, 0.0, grad);

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

            // The cross-fade partner, built ONLY where it is read.
            //
            // It used to be built for every pixel, because it carried its own derivatives and those cannot be
            // branched around. It no longer carries any: the gradient is the same for every cascade (see
            // ShadowPlaneGradient), so the primary's serves, and what is left is a matrix select and a multiply
            // -- which can be skipped. The band is a tenth of a cascade's range by default, so this is work that
            // was being thrown away on the large majority of the screen.
            ShadowProjection partner = primary;
            if (blend) {
                uint pc = min(cascade + 1, count - 1);
                partner = ShadowProjectAt(worldPos + offsetDir * ShadowTexelWorldAt(pc), pc, 0.0, grad);
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
                // Everything that can move a lookup off the exact light ray, in world units. The kernel reaches
                // about three texels from the sample point -- each bilinear quad spans two and sits one texel out
                // -- and the normal offset pushes the sample up to shadow_params.z texels off the surface before
                // any of that. Scaled by the LARGEST cascade's texel, because which cascade this pixel lands in
                // is not decided here for the partner and the unused entries are zero, so the max is both safe
                // and free.
                float texelWorld = max(shadow_texel_world.x, max(shadow_texel_world.y, shadow_texel_world.z));
                float margin = texelWorld * (3.0 + shadow_params.z);
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
                float s = ShadowSample(p, isActor);
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
        // The vertex normal when the draw has one, and a recovered face normal when it does not. Which of
        // those applies is decided per DRAW, not by whether the cel relight is on, and that distinction is
        // the whole point: the attribute arrives zeroed on unlit geometry (see the vbo packing), so its
        // length is the test.
        //
        // Recovering it from the world position's screen derivatives -- two tangents across the pixel quad,
        // crossed -- is right for the room mesh, which is what it was written for. Large flat triangles, a
        // quad sits inside one face, and the normal-offset bias gets something true to push along. It is
        // wrong for a character. Link's mesh is dense enough that most quads straddle a triangle edge,
        // where the derivative of the world position is a step rather than a tangent, so the "normal" jumps
        // between neighbouring quads, the bias pushes the sample somewhere different every few pixels, and
        // the depth comparison flips with it. That is the grey speckle across his skin in shadow, and no
        // amount of bias tuning could have reached it: the input to the bias was noise.
        // Both computed unconditionally and then selected. ddx/ddy are gradient instructions and may not
        // sit inside varying control flow -- the compiler cannot know this particular condition is uniform
        // across every pixel of the draw, and refuses on that basis.
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
        // shadowGrad is the receiver-plane gradient, kept for debug mode 6. It costs nothing to carry:
        // ShadowLitLayers computes it on the way to every comparison regardless.
        float3 shadowGrad;
        float2 shadowLayers = ShadowLitLayers(input.worldPos.xyz, shadowN, input.position.w, shadow_params.x,
                                              input.worldPos.w > 0.5, shadowGrad);
        float shadowLit = min(shadowLayers.x, shadowLayers.y);
        // The filter's own output, before the hardening below rewrites it. Debug mode 5 paints this, and the
        // pair (mode 5 against the shaded picture) is what separates the two halves of the pipeline: a
        // boundary that is already faceted here was drawn by the comparison or by the map, and one that is
        // smooth here and faceted on screen was drawn by the threshold. Those two have no cause in common,
        // and every attempt at this artefact so far has had to guess which it was looking at.
        float shadowCoverage = shadowLit;
        // Harden the edge. What the filter returns is COVERAGE -- how much of the kernel is occluded -- and
        // shading with it directly spreads that ramp across the whole kernel, which is the blur. Remapping
        // it through a narrow ramp centred on half coverage collapses the gradient into an edge instead.
        //
        // This keeps something that simply narrowing the kernel throws away: every tap is bilinear, so
        // coverage varies smoothly BETWEEN texels, and the half-coverage contour is a piecewise-linear curve
        // through the grid rather than a staircase along it. Hardening preserves that sub-texel placement.
        // A narrow ramp rather than a step, so a pixel or two of antialiasing survives on screen.
        //
        // At hardness 0 the lerp weight is 0 and this is exactly the value that came in -- not an
        // approximation of it, which is why the remap is blended in rather than being fed a widening band.
        float shadowHardness = saturate(lerp(shadow_filter.z, shadow_filter.w,
                                             ShadowLadderFraction(input.position.w)));
        // How square-on the surface is to the light. It decides two separate things below -- how far the
        // threshold is allowed to go, and whether the shadow is applied at all -- so it is measured here,
        // before either of them. The light axis is the third column of any cascade's matrix; they share a
        // direction, so cascade 0 will do.
        float3 shadowLightAxis = ShadowLightAxis();
        float shadowIncidence = saturate(abs(dot(shadowN, shadowLightAxis)));
        // Harden in proportion to how well the boundary is sampled, rather than by a fixed amount. The
        // threshold is only as trustworthy as the contour it traces, and that contour degrades with
        // incidence well before it stops carrying information: at sixty degrees the boundary already
        // quantises into steps a couple of world units across, and drawing a hard line through that prints
        // the steps as facets. Tapering the hardening keeps part of the filter's own gradient exactly where
        // the steps live and nowhere else, so the surfaces the hard edge was wanted on -- the ones facing
        // the light, where the boundary is well sampled -- give up nothing.
        //
        // Down to a floor, not to nothing. The threshold clips the penumbra's faint tail and saturates its
        // core, and neither of those needs a hard edge -- both are contrast, and a ramp keeping most of its
        // width still delivers them. Only the last stretch towards an actual step is what traces the texel
        // boundary into facets, so that is the only part the taper takes away.
        shadowHardness *= lerp(shadow_incidence.z, 1.0,
                               smoothstep(shadow_incidence.x, shadow_incidence.y, shadowIncidence));
        // Now remap. What the filter returned is coverage; a narrow ramp centred on half coverage collapses
        // that gradient into an edge. This must come after the taper -- it is the only reader of
        // shadowHardness on the shipping path, so anything written to it below this point would be
        // discarded by the compiler. (Debug mode 8 reads it too, but only inside a branch on a uniform.)
        // Width of the ramp the coverage is remapped through -- and the units it is measured in are the
        // whole question for a cel edge.
        //
        // 0.03 is in COVERAGE, which is not what an eye reads. How many screen pixels that ramp spans
        // depends on how fast coverage happens to change across the surface, and that varies enormously: on
        // a floor square to the light it can be a fraction of a pixel, which aliases into a jagged line,
        // while on a wall raking away from the light the same 0.03 is spread over many pixels and reads as
        // a soft smudge on what was asked to be a hard edge. One number cannot be right for both, because
        // it is not measuring the thing that matters.
        //
        // fwidth(coverage) is how much coverage moves per screen pixel, so dividing into it puts the band in
        // SCREEN units instead: about three quarters of a pixel of ramp everywhere, whatever the surface is
        // doing. That is a hard edge carrying exactly enough antialiasing to not crawl, and it is the same
        // width on the floor and on the wall.
        //
        // Computed unconditionally: fwidth is a gradient instruction and may not sit in varying control
        // flow, the same constraint the recovered normal above is written around.
        float shadowCoverageSlope = fwidth(shadowCoverage);
        // Bounded, and the bounds are the point rather than paranoia. fwidth of a coverage field that is
        // not perfectly smooth is itself noisy: where coverage happens to be locally flat it collapses to
        // nothing and the remap becomes a bare step, which aliases and crawls; where coverage jumps it
        // spikes and the same edge goes soft for a few pixels. Alternating between those two along one
        // boundary is what reads as a scribbled edge rather than a drawn one. The floor keeps a pixel of
        // antialiasing everywhere and the ceiling stops a spike smearing it.
        float shadowBandHard = shadow_plane.w > 0.5 ? clamp(shadowCoverageSlope * 0.75, 0.015, 0.25) : 0.03;
        float shadowBand = lerp(0.5, shadowBandHard, shadowHardness);
        float shadowHard = smoothstep(0.5 - shadowBand, 0.5 + shadowBand, shadowLit);
        shadowLit = lerp(shadowLit, shadowHard, shadowHardness);
        // Debug 2: paint the two caster layers apart instead of shading with them. GREEN where the world
        // layer occludes, RED where the actor layer does. A shadow that vanishes is either coming from a
        // layer that stopped capturing or not being sampled at all, and those look identical once the two
        // are combined -- this is the only way to tell which without guessing.
        // The grazing falloff that used to live here is gone. It stopped applying the shadow at all as a
        // scenery surface turned edge-on to the light, on the argument that the map has no resolution left
        // along the direction such a surface recedes, so its shadow boundary quantises into steps that no
        // bias can reach.
        //
        // The reasoning was sound and the cure was worse. What it bought was a wall keeping a ragged
        // boundary instead of a spiky one; what it cost was that wall losing its shadow outright, and the
        // shadowed-to-unshadowed transition landing in a band forty degrees wide around edge-on -- which on
        // anything curved is a ring of lit surface sitting between the lit part and the shadowed part.
        // Removed at the tuned configuration's request: with the blur widened and the hardening eased, the
        // steps it was hiding no longer read as teeth, and a wall keeping its shadow is worth more than a
        // boundary that is slightly rough.
        //
        // Its measurements survive in shadow_incidence, which now serves only the hardness taper above.
        //
        // DIAGNOSTIC CHANNEL. Modes 1 and 2 show what the shadow system PRODUCED; 3 to 8 show what it was
        // GIVEN. That distinction is the whole reason the rest of this exists.
        //
        // Every knob in shadow_map.h -- the three biases, the incidence band, the hardness ramp, the filter
        // width -- acts on the output of the comparison. Tuning them can make an artefact less visible and
        // can never say where it came from, because a term that is wrong and a term that is right but badly
        // scaled look identical once they have been multiplied into a colour. These views print the inputs
        // instead, unshaded, so the question stops being "does this setting help" and becomes "is this value
        // the shape of the thing on screen".
        //
        // Reading them for a faceted or triangular boundary, in the order that narrows fastest:
        //   4 first. If it is red anywhere the artefact appears, the receiver has no vertex normal and every
        //     downstream term is running off a face normal recovered from screen derivatives -- constant
        //     across a whole triangle, so anything thresholding it prints that triangle's outline exactly.
        //     Dark green is the subtler version: an interpolated normal sagging towards the 1e-4 test, which
        //     flips the shading model INSIDE a triangle rather than between triangles.
        //   3 next, to see it directly. Flat facets of colour on a surface that should be smooth confirm it.
        //   5 against the shaded picture. Faceted here too -> the comparison or the map drew it, and 6 and 7
        //     say which. Smooth here and faceted on screen -> the threshold drew it, and 8 says why.
        //   8 for that: the hardness after the incidence taper. The taper is driven by the normal, so a
        //     per-triangle normal makes a per-triangle hardness, and every triangle then renders the same
        //     shadow boundary with a different edge.
        //   6 for the receiver-plane gradient. Red is the +/-3.2 clamp in ShadowPlaneGradient binding; a
        //     clamp that binds on some triangles of a mesh and not on others is another way to draw one.
        //   7 to rule out cascade selection entirely -- if the shape follows a cascade boundary it is not a
        //     mesh artefact at all.
        //   9 when a knob that should have mattered did not. Red means the kernel has no reach, which makes
        //     the whole plane-bias path inert AND exposes the map's texel staircase directly -- one cause
        //     for both halves of that puzzle.
        //
        // None of these is a shipping path: each replaces the shaded colour outright. `[branch]` because the
        // mode is a uniform, so a build with the channel off pays one jump and none of the arithmetic.
        uint shadowDebugMode = (uint)(shadow_filter.y + 0.5);
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
            // The filter's raw coverage, before the hardening remap. This is the fork: faceted here means
            // the fault is upstream of the threshold, smooth here means the threshold made it.
            texel.rgb = float3(shadowCoverage, shadowCoverage, shadowCoverage);
        } else if (shadowDebugMode == 6) {
            // Receiver-plane gradient. GREEN ramps with its magnitude against the clamp; RED marks pixels
            // where the clamp actually bound and the plane correction stopped being exact. Black past
            // shadow_incidence.w, where no cascade reaches and nothing was projected.
            // The two are exclusive on purpose. Ramping green with the magnitude AND flagging the clamp in
            // red renders a clamped pixel yellow, because a clamped magnitude is by definition the top of
            // the ramp -- so the flag was invisible as a colour of its own, which is the one thing this view
            // exists to show. Green ramps only while the clamp is off; once it binds the pixel is pure red.
            // Measured against the LIVE bound, not against the 3.2 it used to be a literal. Reading the
            // constant here would have made this view lie the moment the bound was swept -- it would still
            // have drawn the old threshold, and the sweep is the whole reason the bound became settable.
            float shadowGradLimit = max(shadow_plane.x, 1e-4);
            float shadowGradMax = max(abs(shadowGrad.x), abs(shadowGrad.y));
            bool shadowGradClamped = shadowGradMax >= shadowGradLimit - (shadowGradLimit * 1e-3);
            texel.rgb = shadowGradClamped ? float3(1.0, 0.0, 0.0)
                                          : float3(0.0, saturate(shadowGradMax / shadowGradLimit), 0.0);
        } else if (shadowDebugMode == 7) {
            // Which cascade this pixel sampled: red, green, blue from nearest to furthest. Cross-fade bands
            // read as the primary cascade's colour, since that is the one the picture is keyed to.
            uint shadowDebugCascade = ShadowCascadeIndex(input.position.w);
            texel.rgb = float3(shadowDebugCascade == 0 ? 1.0 : 0.0, shadowDebugCascade == 1 ? 1.0 : 0.0,
                               shadowDebugCascade == 2 ? 1.0 : 0.0);
        } else if (shadowDebugMode == 9) {
            // How far the PCF kernel actually reaches, in texels: the configured filter width, times
            // whatever the plane bound's taper left of it. GREEN ramps with it; RED means ZERO.
            //
            // This view exists because a null result needed explaining. Sweeping the plane-gradient bound
            // changed nothing, and one thing makes every knob on that path inert at once: a kernel with no
            // reach. At zero the sixteen taps collapse onto one bilinear tap, which softens WITHIN a texel
            // and does nothing about the staircase BETWEEN texels -- so the map's texel grid is drawn
            // straight to the screen, and a staircase crossed at an angle is a row of teeth.
            //
            // It also makes the receiver-plane correction meaningless: that term exists to reconcile taps
            // that sit apart, and taps that sit together have nothing to reconcile. Which is exactly why
            // moving its bound did nothing, and why narrowing an already-collapsed kernel did nothing.
            //
            // Red here and the answer is the filter width, not any bias in this file.
            float shadowReach = min(shadow_filter.x, 1.0) * shadowGrad.z;
            texel.rgb = shadowReach <= 1e-4 ? float3(1.0, 0.0, 0.0) : float3(0.0, saturate(shadowReach), 0.0);
        } else if (shadowDebugMode == 8) {
            // Edge hardness after the incidence taper: RED fully hard, GREEN fully soft. The taper reads the
            // normal, so this is where a per-triangle normal turns into a per-triangle shadow edge.
            texel.rgb = float3(shadowHardness, 1.0 - shadowHardness, 0.0);
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
