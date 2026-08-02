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
        @{update_floats(4)}
    @else
        float3 input@{i + 1} : INPUT@{i};
        @{update_floats(3)}
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
// SamplerComparisonState. Hardware comparison sampling would give a filtered 2x2 per fetch for free, but it
// would not map to the ps_4_0 profile these shaders are compiled against -- the reason the first two
// attempts at this aborted with "cannot map expression to ps_4_0 instruction set". The element type is
// explicit because the fetch has to come back as one float, not a float4 to be truncated.
Texture2DArray<float> g_shadowMap : register(t6);
SamplerState g_shadowSampler : register(s6);

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
    // y = debug. 1 paints everything OUTSIDE a cascade's footprint as fully occluded instead of silently
    //     lit, which is the only way to see where a cascade actually ends. 2 replaces the shading with the
    //     two caster layers separated by colour: green = occluded by the world layer, red = by the actor
    //     layer.
    // z = edge hardness near, w = edge hardness in the furthest cascade, ramped between across the ladder
    //     (see SHADOW_MAP_DEFAULT_EDGE_HARDNESS).
    float4 shadow_filter;
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
float ShadowTap(float2 uvTap, float2 uvCentre, float2 grad, float slice, float z) {
    float stored = g_shadowMap.SampleLevel(g_shadowSampler, float3(uvTap, slice), 0);
    float zAtTap = z + dot(uvTap - uvCentre, grad);
    return zAtTap <= stored ? 1.0 : 0.0; // 1 = this texel does not occlude
}

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
float SampleShadowPCF4(float2 uv, float2 uvCentre, float2 grad, float z, uint cascade, float texelUv,
                       float sliceBase) {
    float slice = sliceBase + (float)cascade;
    // Position in texel space, offset so flooring lands on the lower-left of the surrounding quad.
    float2 texelPos = uv / texelUv - 0.5;
    float2 baseTexel = floor(texelPos);
    float2 subTexel = texelPos - baseTexel;
    float2 uv00 = (baseTexel + 0.5) * texelUv;

    // uvCentre, not uv: every tap in the whole kernel measures its plane correction from the ONE point the
    // receiver's depth was evaluated at, otherwise each 2x2 quad would correct against itself and the
    // sixteen-tap kernel would still disagree with itself across its own width.
    float s00 = ShadowTap(uv00, uvCentre, grad, slice, z);
    float s10 = ShadowTap(uv00 + float2(texelUv, 0.0), uvCentre, grad, slice, z);
    float s01 = ShadowTap(uv00 + float2(0.0, texelUv), uvCentre, grad, slice, z);
    float s11 = ShadowTap(uv00 + float2(texelUv, texelUv), uvCentre, grad, slice, z);

    float top = lerp(s00, s10, subTexel.x);
    float bottom = lerp(s01, s11, subTexel.x);
    return lerp(top, bottom, subTexel.y);
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
float SampleShadowPCF16(float2 uv, float2 grad, float z, uint cascade, float texelUv, float sliceBase) {
    // Spacing is a tunable radius in texels, NOT a free parameter: each bilinear tap already spans a 2x2
    // texel quad, so a radius of one texel puts those quads edge to edge and covers 4x4 contiguously, and
    // anything WIDER leaves texels between the quads sampled by nothing -- a regular hole in the kernel,
    // which reads on screen as a grid laid over the ground. Two texels did exactly that. Below one the
    // quads overlap instead, which only costs redundancy, so this is safe to turn down for a tighter
    // penumbra and must not be turned above 1.0.
    float d = texelUv * min(shadow_filter.x, 1.0);
    float sum = SampleShadowPCF4(uv + float2(-d, -d), uv, grad, z, cascade, texelUv, sliceBase);
    sum += SampleShadowPCF4(uv + float2(d, -d), uv, grad, z, cascade, texelUv, sliceBase);
    sum += SampleShadowPCF4(uv + float2(-d, d), uv, grad, z, cascade, texelUv, sliceBase);
    sum += SampleShadowPCF4(uv + float2(d, d), uv, grad, z, cascade, texelUv, sliceBase);
    return sum * 0.25;
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
    float s = shadow_splits.w;
    if (c == 0) {
        s = shadow_splits.x;
    } else if (c == 1) {
        s = shadow_splits.y;
    } else if (c == 2) {
        s = shadow_splits.z;
    }
    return s;
}

// Compare one cascade. The per-cascade values arrive by value rather than being looked up, which is what
// keeps the caller's selection on literal indices (see the note above). `slice` is only ever a texture
// coordinate, and those may be dynamic.
float ShadowLitCascade(float3 worldPos, float3 normalWs, float4x4 viewProj, float texelWorld, float texelUv,
                       float depthBias, uint slice, float sliceBase) {
    // Push the sample off the surface along its own normal before projecting. A depth-only bias cannot fix
    // curved surfaces -- it only slides the comparison along the light ray, still inside the same polygon
    // -- whereas this moves it sideways, out of the geometry casting onto itself. That is what removes the
    // striped self-shadowing (acne). normalWs always arrives unit length -- the caller resolves the vertex
    // normal or a recovered face normal before this point -- so the term is always live.
    // Single return from a pre-initialized local (see ShadowSplitAt): 1.0 is also the right answer for
    // every rejected case, since a point this cascade cannot see is a point it knows nothing occluding.
    //
    // That silence is exactly what makes a mis-sized cascade impossible to diagnose: a receiver outside the
    // footprint looks identical to one that nothing occludes, so a shadow that stops at the cascade edge
    // reads as a shadow that was never cast. The debug flag inverts the rejected case to "fully occluded",
    // which draws the footprint boundary on screen.
    float lit = shadow_filter.y > 0.5 ? 0.0 : 1.0;
    // Orient the normal to face the light before pushing along it. The offset only helps if it moves the
    // sample OFF the surface towards the light; pushed the other way it drives the sample into the geometry
    // and makes the self-shadowing worse than no offset at all. That sign is not something the caller can
    // guarantee: a normal recovered from screen-space derivatives comes out either way depending on
    // triangle winding and which way the screen's y axis runs, and a vertex normal can disagree with the
    // face it sits on.
    // The light's own axis is the third column of this cascade's matrix (the projection scales the unit z
    // axis by 1/(zFar - zNear)), so the test needs no extra uniform. A surface facing the light has a
    // normal pointing against the direction the light travels.
    float3 lightAxis = normalize(viewProj._13_23_33);
    float3 n = dot(normalWs, lightAxis) > 0.0 ? -normalWs : normalWs;
    // Scale the push by how obliquely the light strikes this surface, sin of the angle between them.
    //
    // Acne is a grazing-angle artefact: depth changes fast across a texel exactly when the surface is
    // nearly edge-on to the light, and hardly at all when it faces the light square. Applying the same
    // push everywhere therefore spent its whole cost where it bought nothing -- on a floor lit from
    // overhead the offset is pure displacement along the light ray, which is peter panning and nothing
    // else, and it is precisely on such floors that a shadow's contact point is most closely read.
    //
    // sin also has the right shape at the other end: it goes to 1 as the surface turns edge-on, which is
    // where the offset stops displacing along the ray at all and starts sliding sideways off the polygon,
    // which is the only thing that actually removes the stripes. So this is not a trade of one artefact
    // for the other -- it moves the whole budget to the angles that need it.
    float ndotl = saturate(dot(n, -lightAxis));
    float grazing = sqrt(saturate(1.0 - (ndotl * ndotl)));
    float3 p = worldPos + n * (shadow_params.z * texelWorld * grazing);
    float4 clip = mul(float4(p, 1.0), viewProj);

    // Projected position and shadow-map coordinate, computed BEFORE any branch on purpose. The screen-space
    // derivatives below have to be taken in flow every pixel of the quad reaches, or neighbouring pixels
    // that took different paths would poison them.
    float safeW = abs(clip.w) > 1e-6 ? clip.w : 1e-6;
    float3 ndc = clip.xyz / safeW;
    // NDC -> texture space (y flips: NDC is +up, textures are +down).
    float2 uv = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    // How fast the receiver's depth changes per unit of shadow-map uv, recovered from screen-space
    // derivatives. The two derivative pairs give depth and uv per screen pixel; inverting the uv Jacobian
    // turns that into depth per uv, which is the receiver plane expressed in the map's own coordinates.
    // ShadowTap uses it to compare each tap against the plane rather than against one point on it.
    float2 duvdx = ddx(uv);
    float2 duvdy = ddy(uv);
    float dzdx = ddx(ndc.z);
    float dzdy = ddy(ndc.z);
    float det = (duvdx.x * duvdy.y) - (duvdx.y * duvdy.x);
    float2 grad = float2(0.0, 0.0);
    if (abs(det) > 1e-12) {
        grad = float2((duvdy.y * dzdx) - (duvdx.y * dzdy), (duvdx.x * dzdy) - (duvdy.x * dzdx)) / det;
    }
    // Bound it. At a silhouette the quad straddles two surfaces and the derivative is meaningless, and an
    // unbounded correction there would punch a hole through the shadow. The depth range of every cascade is
    // five times its radius by construction and a texel is two radii over the resolution, so a surface at
    // forty-five degrees to the light has a gradient of 2/5 -- the radius and the resolution both cancel.
    // 3.2 is eight times that, about eighty-three degrees, past which a receiver is edge-on enough that the
    // constant and normal-offset terms are the right tools.
    grad = clamp(grad, -3.2, 3.2);

    if (clip.w > 0.0 && all(abs(ndc.xy) <= 1.0) && ndc.z >= 0.0 && ndc.z <= 1.0) {
        lit = SampleShadowPCF16(uv, grad, ndc.z - depthBias, slice, texelUv, sliceBase);
    }
    return lit;
}

// Dispatch to one cascade with literal indices. The chain covers SHADOW_MAP_MAX_CASCADES entries; if that
// ever grows, this grows with it.
//
// It SELECTS the cascade's constants and then samples once, rather than branching around four separate
// calls to ShadowLitCascade. That distinction is the difference between a shader that compiles in
// milliseconds and one that does not. ShadowLitCascade has no callable form -- ps_4_0 inlines everything --
// so a four-arm dispatch pasted its sixteen-tap filter in four times; ShadowLit calls this twice (the
// cascade and its cross-fade partner) and PSMain calls ShadowLit twice (the world layer and the actor
// layer), which multiplied out to 256 inlined texture-fetch sites in every receiver shader. FXC at
// optimisation level 2 takes a long time over that, and it runs SYNCHRONOUSLY inside a frame the first time
// each material is drawn -- which is exactly the hitch felt as new geometry rotates into view. Selecting
// first cuts it to 64 sites with identical output: only one arm's fetches ever executed anyway.
//
// Each branch moves four registers' worth of constants, so there is nothing left worth a real branch;
// flattening to conditional moves is cheaper than the jump. Every index stays literal -- see ShadowSplitAt
// for why a computed one cannot be used here.
float ShadowLitAt(float3 worldPos, float3 normalWs, uint cascade, float sliceBase) {
    float4x4 viewProj = shadow_view_proj[0];
    float texelWorld = shadow_texel_world.x;
    float texelUv = shadow_texel_uv.x;
    float depthBias = shadow_depth_bias.x;
    if (cascade == 1) {
        viewProj = shadow_view_proj[1];
        texelWorld = shadow_texel_world.y;
        texelUv = shadow_texel_uv.y;
        depthBias = shadow_depth_bias.y;
    } else if (cascade == 2) {
        viewProj = shadow_view_proj[2];
        texelWorld = shadow_texel_world.z;
        texelUv = shadow_texel_uv.z;
        depthBias = shadow_depth_bias.z;
    } else if (cascade == 3) {
        viewProj = shadow_view_proj[3];
        texelWorld = shadow_texel_world.w;
        texelUv = shadow_texel_uv.w;
        depthBias = shadow_depth_bias.w;
    }
    // `slice` is only ever a texture coordinate, and those may be dynamic -- see ShadowLitCascade.
    return ShadowLitCascade(worldPos, normalWs, viewProj, texelWorld, texelUv, depthBias, cascade, sliceBase);
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
        if (viewDepth > shadow_splits.z) {
            step = 3.0;
        }
        band = saturate(step / (float)(count - 1));
    }
    return band;
}

// Pick a cascade by view distance and cross-fade into the next one over the last slice of the range.
// Without the fade the resolution change shows up as a hard line sweeping across the ground as the camera
// moves -- "cascade popping". smoothstep rather than a linear ramp so the seam has no visible corner.
float ShadowLit(float3 worldPos, float3 normalWs, float viewDepth, float sliceBase) {
    // Single return, pre-initialized to "fully lit" -- which is also the answer when no cascades were
    // rendered this frame (count == 0).
    float lit = 1.0;
    uint count = (uint)shadow_params.x;
    if (count > 0) {
        // First cascade whose far split still covers this depth; the last one catches everything beyond.
        uint cascade = count - 1;
        if (viewDepth <= shadow_splits.x) {
            cascade = 0;
        } else if (count > 1 && viewDepth <= shadow_splits.y) {
            cascade = 1;
        } else if (count > 2 && viewDepth <= shadow_splits.z) {
            cascade = 2;
        }
        cascade = min(cascade, count - 1);

        lit = ShadowLitAt(worldPos, normalWs, cascade, sliceBase);

        // Cross-fade band at the far edge of this cascade, where the next one also covers the point.
        // Sampling both and blending is what hides the resolution change; a hard switch draws a visible
        // line that sweeps across the ground as the camera moves.
        if (cascade + 1 < count) {
            // Not named `far`/`near`: those are legacy Windows macros, and this source is compiled by name
            // at runtime where a stray definition would be baffling to debug.
            float farEdge = ShadowSplitAt(cascade);
            float nearEdge = (cascade == 0) ? 0.0 : ShadowSplitAt(cascade - 1);
            float bandStart = farEdge - (farEdge - nearEdge) * shadow_params.y;
            if (viewDepth > bandStart) {
                float t = smoothstep(bandStart, farEdge, viewDepth);
                lit = lerp(lit, ShadowLitAt(worldPos, normalWs, cascade + 1, sliceBase), t);
            }
        }
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
        float shadowWorldLit = ShadowLit(input.worldPos.xyz, shadowN, input.position.w, 0.0);
        // Scenery also takes the actor caster layer; a character does not, so it is never shadowed by
        // another character or by itself. [branch] because the value is constant across a draw call, so the
        // character case genuinely skips the second set of taps rather than computing and discarding them.
        float shadowActorLit = 1.0;
        [branch]
        if (input.worldPos.w > 0.5) {
            shadowActorLit = ShadowLit(input.worldPos.xyz, shadowN, input.position.w, shadow_params.x);
        }
        float shadowLit = min(shadowWorldLit, shadowActorLit);
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
        float3 shadowLightAxis = normalize(shadow_view_proj[0]._13_23_33);
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
        shadowHardness *= lerp(@{o_shadow_min_hardness_scale}, 1.0,
                               smoothstep(@{o_shadow_min_incidence}, @{o_shadow_full_incidence},
                                          shadowIncidence));
        // Now remap. What the filter returned is coverage; a narrow ramp centred on half coverage collapses
        // that gradient into an edge. This must come after the taper -- it is the only reader of
        // shadowHardness, so anything written to it below this point would be discarded by the compiler.
        float shadowBand = lerp(0.5, 0.03, shadowHardness);
        float shadowHard = smoothstep(0.5 - shadowBand, 0.5 + shadowBand, shadowLit);
        shadowLit = lerp(shadowLit, shadowHard, shadowHardness);
        // Debug 2: paint the two caster layers apart instead of shading with them. GREEN where the world
        // layer occludes, RED where the actor layer does. A shadow that vanishes is either coming from a
        // layer that stopped capturing or not being sampled at all, and those look identical once the two
        // are combined -- this is the only way to tell which without guessing.
        // Stop applying the shadow at all as the surface turns edge-on to the light.
        //
        // Not a softening: this declines to use a number that carries no information. On a surface nearly
        // parallel to the light the map has almost no resolution along the direction that surface recedes,
        // so the shadow BOUNDARY quantises into steps of one texel divided by the sine of the angle -- under
        // two world units at sixty degrees, seventeen at five, forty-two at two, which is a character's
        // whole height. Those steps are the teeth, and no bias touches them because nothing is being
        // mis-compared: the boundary is being drawn at a resolution that does not exist.
        //
        // Fading it out is what the physics says anyway. A surface edge-on to the light receives almost no
        // light and can therefore carry almost no shadow, so illumination and map resolution reach zero
        // together and the term stops mattering exactly where it stops being computable.
        float shadowApply = smoothstep(0.0, @{o_shadow_min_incidence}, shadowIncidence);
        [branch]
        if (shadow_filter.y > 1.5) {
            texel.rgb = float3(1.0 - shadowActorLit, 1.0 - shadowWorldLit, 0.0);
        } else {
            texel.rgb *= lerp(1.0, lerp(1.0 - shadow_params.w, 1.0, shadowLit), shadowApply);
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
        @if(o_alpha_threshold)
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
