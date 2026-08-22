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
// A plain sampler, compared by hand in SampleShadowPCF4, NOT a SamplerComparisonState.
//
// The hardware comparison unit would do the fetch, the four compares and the blend in one instruction, and
// that is genuinely what it is for -- but comparison sampling of a texture ARRAY is a Shader Model 4.1
// feature, and this renderer accepts adapters down to feature level 10_0. Compiled at ps_4_0 it is
// "error X4532: cannot map expression to ps_4_0 instruction set", which surfaces as a failed shader compile
// in the middle of a frame and takes the process down. It was tried and reverted for exactly that.
//
// The element type is explicit because the fetch has to come back as one float, not a float4 to be
// truncated.
Texture2DArray<float> g_shadowMap : register(t6);
SamplerState g_shadowSampler : register(s6);

// SOH [Enhancement] The ACTOR caster layer, which may live in an array of its own at its own resolution
// (see fast/shadow_map.h). While the two layers are the same size the backend binds the SAME view here, so
// every fetch below reads exactly the texels it read before the layers could be sized apart.
//
// Selected with a branch rather than by duplicating the kernel: `isActor` is uniform for the whole of a
// lookup, both textures are referenced statically, and no index is dynamic -- so this stays inside what
// ps_4_0 will map.
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
    //     5 the filter's raw coverage, BEFORE the cel threshold and the incidence weight rewrite it
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
}

// Bilinear PCF over the four texels surrounding the sample point: fetch each, compare each, then weight the
// RESULTS by the sub-texel position. Comparing first and filtering after is the whole point -- filtering the
// stored depths and comparing once would give a wrong penumbra.
//
// Done by hand because the hardware unit that does exactly this cannot be reached here: see the sampler
// declaration above for why a SamplerComparisonState is not an option at ps_4_0 on a texture array.
//
// The sampler filters point-wise on purpose, which is the other half of that ordering: a linear sampler
// would blend the stored depths before this code ever compared them.
//
// The offsets have to be a full texel apart and land on texel centres. An earlier version kept half-texel
// quincunx offsets that suited a comparison sampler, where every fetch already straddles four texels;
// against a point sampler those four taps usually land inside the SAME texel, return the same value, and
// average to exactly one hard sample -- no filtering at all, which is what made edges stair-step.
float SampleShadowPCF4(float2 uv, float z, float slice, float texelUv, bool isActor) {
    // Position in texel space, offset so flooring lands on the lower-left of the surrounding quad.
    float2 texelPos = uv / texelUv - 0.5;
    float2 baseTexel = floor(texelPos);
    float2 subTexel = texelPos - baseTexel;
    float2 uv00 = (baseTexel + 0.5) * texelUv;

    // Component order is Gather's: w is the texel at (0,0) from uv00, z is (1,0), x is (0,1), y is (1,1).
    // Holding all four as one float4 is the point of this shape -- the compare and the blend are then single
    // vector instructions instead of four scalar comparisons and three lerps.
    //
    // Selected with a branch, not a ternary: ps_4_0 encodes the texture and sampler slots into the sample
    // instruction, so choosing between two (texture, sampler) pairs has to be flow control.
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

    // step(a, b) is b >= a, so this is "the receiver is at or in front of the stored depth" -- 1 where the
    // texel does not occlude -- for all four at once. One reference depth for the whole quad: the offset
    // that keeps a surface from shadowing itself is applied by the rasterizer during the depth pass, not
    // here (see SHADOW_MAP_SLOPE_BIAS).
    float4 lit = step(z, stored);

    // Bilinear weights, written out. lerp(lerp(w, z, sx), lerp(x, y, sx), sy) is exactly this sum, and as a
    // dot it is one instruction instead of three dependent ones.
    float2 inv = 1.0 - subTexel;
    float4 weights = float4(inv.x * subTexel.y, subTexel.x * subTexel.y, subTexel.x * inv.y, inv.x * inv.y);
    return dot(lit, weights);
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
// The depth offset that keeps a surface from shadowing itself, measured from the ANGLE between the light and
// this pixel's normal.
//
// Acne is the receiver's depth being compared against a stored depth that was quantised over a whole texel.
// How far the receiver's own depth travels across that texel is not a constant -- it is the texel's width
// times the tangent of the angle between the surface and the light. Facing the light square on, the depth
// barely moves and almost nothing is needed; edge-on it runs away, and so must the offset. A fixed number
// has to be large enough for the worst angle in the scene and is then far too large everywhere else, which
// is peter panning bought for nothing.
//
// Scaled by the CASCADE'S OWN texel, so the same expression means the right physical distance in each of
// them: a near-cascade texel is a fraction of a world unit and a far one is several. Nothing here is a
// magic number in world units -- change the resolution or the split distances and this follows.
//
// The MAGNITUDE of the incidence, not the signed dot. A face normal recovered from screen derivatives points
// either way depending on winding and which way screen y runs, and the signed form would read a surface
// facing away from the light as one facing it -- handing the largest offset to the surfaces that need the
// least. The magnitude is the same on both sides and only grows where the surface is genuinely tangent.
//
// Two clamps, both load-bearing. The floor is what covers the flat case, where the angle term goes to zero
// but the depth buffer's own rounding does not -- a D16 map quantises every stored depth whatever the
// polygon is doing. The ceiling stops the tangent running to infinity at true tangency, where it would
// push the comparison so far that the surface stops receiving any shadow at all.
float ShadowDepthBias(float3 normalWs, float3 lightAxis, float texelWorld, float depthScale) {
    float ndotl = saturate(abs(dot(normalWs, lightAxis)));
    float sinTheta = sqrt(saturate(1.0 - (ndotl * ndotl)));
    float tanTheta = sinTheta / max(ndotl, 0.1); // 0.1 caps the tangent near ten, about 84 degrees
    // In texels, then into this cascade's depth units. texelWorld is one texel as a world distance and
    // depthScale is world-to-NDC-depth, so their product is one texel expressed as depth.
    float biasTexels = min(1.0 + tanTheta, 8.0);

    // Capped in WORLD units as well as in texels, and this second cap is the one that matters.
    //
    // Measuring the offset in texels is right -- acne appears at the scale of the map's own grid, so the
    // offset has to track that grid. What it must not do is track it all the way out. A texel of the near
    // cascade is a fifth of a world unit and a texel of the far one is nearly four, so the same eight texels
    // are 1.6 units up close and 29 at distance -- and Link is about sixty units tall. An offset of 29 units
    // does not bias a shadow, it deletes it: everything shorter than that along the light stops casting
    // entirely, and what remains is eaten into. That is a shadow fading out with distance, which is exactly
    // what it looked like.
    //
    // Three units is invisible as panning at this scale -- a twentieth of Link's height -- and it is the
    // same ceiling the normal-offset term used to carry for the same reason. The cost is that the far
    // cascade gets less than one texel of offset where the texel formula asked for eight, so acne can come
    // back there; a shadow that is slightly speckled at distance is worth more than no shadow at all.
    return min(biasTexels * texelWorld, 3.0) * depthScale;
}

ShadowProjection ShadowProject(float3 p, float3 normalWs, float3 lightAxis, float4x4 viewProj, float texelUv,
                               float texelWorld, uint cascade, float sliceBase) {
    float4 clip = mul(float4(p, 1.0), viewProj);

    float safeW = abs(clip.w) > 1e-6 ? clip.w : 1e-6;
    float3 ndc = clip.xyz / safeW;
    // NDC -> texture space (y flips: NDC is +up, textures are +down).
    float2 uv = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    // World-to-NDC-depth for THIS cascade, read off its own matrix rather than plumbed in: the projection
    // scales the light's unit z axis by 1/(zFar - zNear), so that column's length is exactly that factor.
    float depthScale = length(viewProj._13_23_33);

    ShadowProjection o;
    o.uv = uv;
    o.z = ndc.z - ShadowDepthBias(normalWs, lightAxis, texelWorld, depthScale);
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
// The filter proper: a grid of bilinear taps, weighted smoothly, stretched along the direction the receiver
// runs away from the light.
//
// THREE things are being asked of one kernel, and they are not three kernels.
//
// 1. Width. One bilinear tap softens WITHIN a texel and does nothing about the staircase BETWEEN texels, so
//    the map's grid arrives on screen as stair-stepping. A grid of taps one texel apart spreads the step
//    over enough pixels to stop reading as a staircase.
//
// 2. Smooth weights, not a flat average. A flat average is a BOX, and a box has an abrupt end: convolved
//    with a shadow boundary it does make a ramp, but the ramp's slope JUMPS where the box's edge crosses the
//    boundary, so as the surface varies by a texel the half-coverage contour does not slide, it hops.
//    Threshold that for a cel edge and every hop is a kink -- the line comes out scribbled rather than
//    drawn. That is not a hypothetical; it is what the previous attempt at this looked like. The weight
//    below goes to zero smoothly at the rim AND has zero slope there, so no tap enters or leaves the sum
//    abruptly and the contour moves continuously. It costs three arithmetic operations per tap.
//
// 3. Projective aliasing. On a surface turning edge-on to the light the map has almost no resolution along
//    the direction the surface recedes, so the boundary quantises into steps of one texel over the sine of
//    the angle -- eight texels per step at eighty-three degrees, twenty at eighty-seven. No bias reaches
//    those, because nothing is being mis-compared: the boundary is being drawn at a resolution that does not
//    exist. What CAN be done is average over a whole step instead of resolving it, which turns a hard
//    staircase into a soft gradient in the same direction -- the same missing information, presented as
//    penumbra instead of saw-teeth. So the grid's spacing ALONG that direction stretches and its spacing
//    ACROSS it does not, because across it the map samples perfectly well and widening would only smear an
//    edge that was already right.
//
// `stretch` is continuous and the tap COUNT is fixed. Deriving a count per pixel is what printed mesh
// triangles onto the screen last time: the stretch comes from a planar function, so it is constant across a
// triangle, and rounding a per-triangle constant to an integer thresholds it -- which draws that triangle's
// outline exactly. A clamp leaves the value continuous where a round does not; only the slope kinks, and a
// kink does not draw an outline.
//
// `[loop]`, not `[unroll]`. Unrolled, the whole quad appears nine times in the compiled shader, and this
// file is compiled by FXC INSIDE a frame the first time each material draws -- its size is the hitch felt
// when new geometry rotates into view.
float SampleShadowPCF(float2 uv, float z, float slice, float texelUv, bool isActor, float2 axis, float stretch) {
    float2 across = float2(-axis.y, axis.x);
    float sum = 0.0;
    float weightSum = 0.0;
    [loop]
    for (uint i = 0; i < 9; i++) {
        // -1, 0, +1 on each side of the sample point.
        float a = (float)(i % 3) - 1.0;
        float b = (float)(i / 3) - 1.0;
        // One texel across, `stretch` texels along.
        float2 offset = ((axis * (a * stretch)) + (across * b)) * texelUv;
        // Smooth compact weight, (1 - r^2)^2 -- zero AND flat where it reaches zero, which is what keeps the
        // half-coverage contour sliding instead of hopping.
        //
        // The 0.25 is the whole kernel. It normalises so that the FURTHEST taps in the grid, the four
        // corners at a^2 + b^2 == 2, land at r^2 = 0.5 and still carry a quarter of the centre's weight; the
        // falloff reaches zero at r^2 == 1, one ring outside the grid, where there is no tap to receive it.
        //
        // It was 0.5, and that put the corners exactly ON the zero. Four of the nine taps were fetched and
        // then multiplied by nothing, and the centre alone carried half the total -- a nine-tap kernel that
        // sampled like a five-tap one and blurred like barely more than a single tap. The threshold below
        // then squeezed what little remained back out, and the result on screen was indistinguishable from
        // no filter at all. Correct now: 23.5 percent at the centre, nothing wasted.
        float r2 = saturate(((a * a) + (b * b)) * 0.25);
        float w = 1.0 - r2;
        w *= w;
        sum += w * SampleShadowPCF4(uv + offset, z, slice, texelUv, isActor);
        weightSum += w;
    }
    return sum / max(weightSum, 1e-6);
}

float ShadowSample(ShadowProjection p, bool isActor, float2 axis, float stretch) {
    float lit = abs(shadow_range.y - 1.0) < 0.5 ? 0.0 : 1.0;
    if (p.inside > 0.5) {
        lit = SampleShadowPCF(p.uv, p.z, p.slice, p.texelUv, isActor, axis, stretch);
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

ShadowProjection ShadowProjectAt(float3 p, float3 normalWs, float3 lightAxis, uint cascade, float sliceBase) {
    float4x4 viewProj = shadow_view_proj[0];
    float texelUv = shadow_texel_uv.x;
    if (cascade == 1) {
        viewProj = shadow_view_proj[1];
        texelUv = shadow_texel_uv.y;
    } else if (cascade == 2) {
        viewProj = shadow_view_proj[2];
        texelUv = shadow_texel_uv.z;
    }
    return ShadowProject(p, normalWs, lightAxis, viewProj, texelUv, ShadowTexelWorldAt(cascade), cascade, sliceBase);
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
// cascade choice and the matrix multiply are functions of the receiver, not of which layer is being asked
// about -- so computing them twice was computing them twice identically. Layer L, cascade C is slice
// L*count + C, and that is the whole of the difference: the actor lookup is the world lookup with the layer
// stride added to its slice.
//
// `wantActors` is the receiver kind, constant across a draw call, and it gates only the fetches -- never the
// projection, which has to run for the world layer regardless. So a character pays nothing for the actor
// half it skips, exactly as before, while scenery stops paying twice for the half they share.
float2 ShadowLitLayers(float3 worldPos, float3 normalWs, float viewDepth, float layerStride,
                       bool wantActors) {
    // Single return, pre-initialized to "fully lit" -- which is also the answer when no cascades were
    // rendered this frame (count == 0), and for the actor layer whenever this receiver does not take it.
    float2 lit = float2(1.0, 1.0);
    uint count = (uint)shadow_params.x;
    if (count > 0) {
        uint cascade = ShadowCascadeIndex(viewDepth);
        // The direction the light travels, shared by every cascade. Taken once here: the depth bias reads it
        // for every projection this pixel makes, and the actor slab test below reads it again.
        float3 lightAxis = ShadowLightAxis();

        // Which way, on the map, this receiver runs away from the light -- and how fast.
        //
        // Taken from the receiver's NORMAL and the light's own basis, not from screen-space derivatives of
        // the projected position. That is not a shortcut, it is the more correct of the two. A derivative
        // reads across the pixel quad, so at a silhouette -- where the quad straddles two surfaces -- it
        // measures the step between them and returns a direction belonging to neither. The plane's slope in
        // light space is exact wherever the normal is: for a plane with normal N, depth varies with the two
        // lateral axes as -(N.lx)/(N.lz) and -(N.ly)/(N.lz), so the direction of steepest recede is just
        // (N.lx, N.ly) and its magnitude is the tangent of the angle to the light.
        //
        // It is also the same tangent the depth bias uses (see ShadowDepthBias), which is not a coincidence:
        // how far the receiver's depth travels across a texel and how far the kernel must reach to average
        // over a quantisation step are the same geometry asked two ways.
        //
        // The y flip is the projection's: uv is (ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5), so the light's +y
        // is the map's -y. Getting this backwards would stretch the kernel across the step instead of along
        // it -- smearing the edge that was already correct and leaving the one that was not.
        float2 shadowAxis = float2(1.0, 0.0);
        float shadowStretch = 1.0;
        {
            float3 lightX = normalize(shadow_view_proj[0]._11_21_31);
            float3 lightY = normalize(shadow_view_proj[0]._12_22_32);
            float2 nxy = float2(dot(normalWs, lightX), dot(normalWs, lightY));
            float lateral = length(nxy);
            if (lateral > 1e-5) {
                shadowAxis = float2(nxy.x, -nxy.y) / lateral;
            }
            // Clamped, not rounded, and bounded at four: past that the kernel is averaging over more ground
            // than the shadow it is trying to draw, and a wall keeping a soft boundary is worth more than one
            // whose shadow has been smeared into nothing.
            float ndotl = saturate(abs(dot(normalWs, lightAxis)));
            shadowStretch = clamp(lateral / max(ndotl, 0.1), 1.0, 4.0);
        }

        // Past the furthest point any cascade's footprint reaches, there is nothing to look up and the
        // answer is already the one `lit` holds. The bound is computed where the cascades are built, from
        // the boxes themselves rather than from the split ladder -- a cascade's box overshoots its band by a
        // long way, and cutting at the split would take real shadows with it (a low sun throws them well
        // past the band that cast them).
        if (viewDepth <= shadow_range.x) {
            ShadowProjection primary = ShadowProjectAt(worldPos, normalWs, lightAxis, cascade, 0.0);

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
                partner = ShadowProjectAt(worldPos, normalWs, lightAxis, pc, 0.0);
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
                    // do not care how big the map is -- but the kernel walks in TEXELS, and that layer may
                    // have its own. Equal to the world layer's whenever the two are the same size.
                    p.texelUv = ShadowActorTexelUvAt(isPartner ? min(cascade + 1, count - 1) : cascade);
                }
                float s = ShadowSample(p, isActor, shadowAxis, shadowStretch);
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
        float2 shadowLayers = ShadowLitLayers(input.worldPos.xyz, shadowN, input.position.w, shadow_params.x,
                                              input.worldPos.w > 0.5);
        float shadowLit = min(shadowLayers.x, shadowLayers.y);
        // What the filter returned is COVERAGE -- what fraction of the kernel is occluded -- and debug mode 5
        // prints exactly this, before anything below rewrites it.
        float shadowCoverage = shadowLit;

        // Collapse that gradient back into an edge.
        //
        // Shading with coverage directly spreads the filter's whole ramp across the picture, which is the
        // blur. Remapping it through a narrow ramp centred on half coverage turns it back into a boundary --
        // and crucially it keeps what widening the filter bought: every tap is bilinear and the taps are
        // weighted smoothly, so coverage varies continuously BETWEEN texels and the half-coverage contour is
        // a smooth curve through the grid rather than a staircase along it. The threshold reads that
        // sub-texel placement. Widen and then threshold is not a round trip; it is how a hard edge gets to
        // be placed more finely than the map's own grid.
        //
        // A ramp rather than a step, so a pixel or two of antialiasing survives and the line does not crawl
        // as the camera moves. 0.20 is in coverage units -- about two fifths of the filter's transition, so
        // roughly a texel and a half of softness is kept rather than the whole width being squeezed back
        // out. Threshold too narrowly and the widening buys nothing visible: the edge returns to the map's
        // own grid and stair-steps exactly as it did before, which is the failure this number decides.
        //
        // This is also where the previous attempt went wrong, and it was not the threshold's fault. It was
        // being fed a BOX-filtered coverage, whose contour hops rather than slides, and a threshold on a
        // hopping contour draws a scribbled line. The kernel above is smooth for this reason.
        shadowLit = smoothstep(0.5 - 0.20, 0.5 + 0.20, shadowLit);

        // How square-on this receiver is to the light, which is what the shadow's darkness is weighted by.
        //
        // This is the article's own remedy for projective aliasing, and it is the only one it offers beyond
        // "do the perspective-aliasing techniques": "Projective aliasing occurs when the surface normal is
        // orthogonal to the light; these surfaces should be receiving less light based on diffuse lighting
        // equations."
        //
        // The reasoning is that the artefact hides itself. Teeth appear where the surface runs parallel to
        // the light rays, because that is where the map has no resolution along the direction the surface
        // recedes -- and a surface parallel to the light rays is a surface receiving almost no light from it,
        // so there should be nothing there to draw a hard boundary ON. It hides nothing here because nothing
        // in this path carries a diffuse term: the cel relight is objects-only, so the static scene is albedo
        // times shadow, and the relight itself is half-Lambert, which maps an orthogonal normal to the MIDDLE
        // of its ramp rather than to zero. Both of those deliberately remove the falloff the argument needs.
        //
        // So the weight is applied here instead, to the shadow rather than to the light. Same shape, one
        // multiply.
        //
        // The MAGNITUDE of the incidence, not the signed dot. The receiver's normal cannot be trusted to
        // point outward -- a face normal recovered from screen derivatives comes out either way depending on
        // winding and which way screen y runs -- and the signed form would then read a surface facing away
        // from the light as one facing it. The magnitude is the same on both sides and only collapses where
        // the surface is genuinely tangent, which is exactly the band the teeth live in.
        //
        // No threshold and no band, which is what separates this from the incidence taper that used to live
        // here and was removed. That one smoothstepped between two tunable bounds about twenty and sixty
        // degrees apart, and a falloff that wide put a ring of unshadowed surface around anything curved.
        // cos itself is 0.71 at forty-five degrees and 0.05 at eighty-seven: it barely touches a surface that
        // is lit at all, and only empties out at true tangency.
        float shadowIncidence = saturate(abs(dot(shadowN, ShadowLightAxis())));
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
        //   5 to see what the filter returned, before anything remaps it. This is the fork. A boundary that
        //     is already faceted HERE was drawn by the comparison or by the map itself -- its resolution, the
        //     matrix it was drawn with, or the depth pass's bias -- and no work downstream will reach it. One
        //     that is smooth here and hard-edged badly on screen was drawn by the threshold.
        //
        //     TWO things sit between this view and the shaded picture, and neither has a view of its own: the
        //     cel threshold, which remaps coverage through a narrow ramp centred on half, and the incidence
        //     weight, which scales the shadow's darkness by how square-on the surface is to the light. So
        //     where the shaded picture has LESS shadow than this view, suspect the weight and a surface near
        //     tangent to the light; where its boundary is sharper or kinkier, suspect the threshold.
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
            // The coverage the depth comparison produced, raw. The shaded picture scales this by the
            // incidence weight, so the two differ where a surface is near tangent to the light -- see the
            // reading order above.
            texel.rgb = float3(shadowCoverage, shadowCoverage, shadowCoverage);
        } else if (shadowDebugMode == 7) {
            // Which cascade this pixel sampled: red, green, blue from nearest to furthest. Cross-fade bands
            // read as the primary cascade's colour, since that is the one the picture is keyed to.
            uint shadowDebugCascade = ShadowCascadeIndex(input.position.w);
            texel.rgb = float3(shadowDebugCascade == 0 ? 1.0 : 0.0, shadowDebugCascade == 1 ? 1.0 : 0.0,
                               shadowDebugCascade == 2 ? 1.0 : 0.0);
        } else {
            texel.rgb *= lerp(1.0 - (shadow_params.w * shadowIncidence), 1.0, shadowLit);
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
