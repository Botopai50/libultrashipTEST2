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
@if(o_toon)
float3 normal : NORMAL;
@{update_floats(3)}
@end
@if(o_shadow_map)
float3 worldPos : WORLDPOS;
@{update_floats(3)}
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
Texture2DArray g_shadowMap : register(t6);
SamplerComparisonState g_shadowSampler : register(s6);

// Everything is float4-shaped on purpose: HLSL gives each element of a `float arr[n]` its own 16-byte
// register, so a scalar array would waste three quarters of its space and make the C++ layout easy to get
// subtly wrong. Layout matches the PerShadowCB C++ struct exactly.
cbuffer PerShadowCB : register(b3) {
    row_major float4x4 shadow_view_proj[@{o_shadow_max_cascades}];
    float4 shadow_splits;      // far distance of each cascade, world units
    float4 shadow_texel_world; // world size of one texel, per cascade
    float4 shadow_texel_uv;    // one texel in UV terms (1/resolution), per cascade
    // x = active cascade count (0 = no shadow map this frame), y = cross-fade band as a fraction of the
    // cascade, z = receiver push along the normal in texels, w = darkness where fully occluded.
    float4 shadow_params;
}

// Four taps in a quincunx around the centre. Each tap is a hardware comparison fetch, and with a linear
// comparison sampler every fetch is itself a filtered 2x2 -- so these four cover a 4x4 neighbourhood for
// the cost of four samples. That is the whole reason for a comparison sampler over a plain depth read.
float SampleShadowPCF4(float2 uv, float z, uint cascade, float texelUv) {
    const float2 offsets[4] = {
        float2(-0.5, -0.5), float2(0.5, -0.5),
        float2(-0.5, 0.5), float2(0.5, 0.5)
    };
    float sum = 0.0;
    [unroll]
    for (int i = 0; i < 4; i++) {
        sum += g_shadowMap.SampleCmpLevelZero(g_shadowSampler,
                                              float3(uv + offsets[i] * texelUv, cascade), z);
    }
    return sum * 0.25;
}

// Project into one cascade and return how lit that cascade says this point is (1 = lit, 0 = occluded).
// Outside the cascade's footprint there is nothing to occlude, so the answer is "lit" -- which is also
// what the border-clamped sampler returns, but checking here avoids the fetch entirely.
float ShadowLitFromCascade(float3 worldPos, float3 normalWs, uint cascade) {
    // Push the sample off the surface along its own normal before projecting. A depth-only bias cannot fix
    // curved surfaces -- it only slides the comparison along the light ray, still inside the same polygon
    // -- whereas this moves it sideways, out of the geometry casting onto itself. That is what removes the
    // striped self-shadowing (acne). normalWs is zero for receivers that carry no normal (the toon variant
    // is off), and then this term simply vanishes and the rasterizer's slope bias carries it alone.
    float3 p = worldPos + normalWs * (shadow_params.z * shadow_texel_world[cascade]);

    float4 clip = mul(float4(p, 1.0), shadow_view_proj[cascade]);
    if (clip.w <= 0.0) {
        return 1.0;
    }
    float3 ndc = clip.xyz / clip.w;
    if (any(abs(ndc.xy) > 1.0) || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0; // outside this cascade's footprint: nothing here is known to occlude
    }
    // NDC -> texture space (y flips: NDC is +up, textures are +down).
    float2 uv = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    return SampleShadowPCF4(uv, ndc.z, cascade, shadow_texel_uv[cascade]);
}

// Pick a cascade by view distance and cross-fade into the next one over the last slice of the range.
// Without the fade the resolution change shows up as a hard line sweeping across the ground as the camera
// moves -- "cascade popping". smoothstep rather than a linear ramp so the seam has no visible corner.
float ShadowLit(float3 worldPos, float3 normalWs, float viewDepth) {
    uint count = (uint)shadow_params.x;
    if (count == 0) {
        return 1.0; // no cascades were rendered this frame
    }

    uint cascade = count - 1;
    [unroll]
    for (uint i = 0; i < 4; i++) {
        if (i < count && viewDepth <= shadow_splits[i]) {
            cascade = min(cascade, i);
        }
    }

    float lit = ShadowLitFromCascade(worldPos, normalWs, cascade);

    // Cross-fade band at the far edge of this cascade, where the next one also covers the point. Sampling
    // both and blending is what hides the resolution change; a hard switch draws a visible line that
    // sweeps across the ground as the camera moves.
    if (cascade + 1 < count) {
        float nearEdge = (cascade == 0) ? 0.0 : shadow_splits[cascade - 1];
        float span = shadow_splits[cascade] - nearEdge;
        float bandStart = shadow_splits[cascade] - span * shadow_params.y;
        if (viewDepth > bandStart) {
            float t = smoothstep(bandStart, shadow_splits[cascade], viewDepth);
            lit = lerp(lit, ShadowLitFromCascade(worldPos, normalWs, cascade + 1), t);
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
@if(o_toon)
    , float3 normal : NORMAL
@end
@if(o_shadow_map)
    , float3 worldPos : WORLDPOS
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

    @if(o_toon)
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
        @if(o_toon)
            float3 shadowN = normalize(input.normal);
        @else
            // No normal on this draw, so no normal-offset push -- the rasterizer's slope-scaled bias is
            // the only thing keeping this surface off its own depth values.
            float3 shadowN = float3(0.0, 0.0, 0.0);
        @end
        // input.position.w is the clip-space w the rasterizer interpolated, which for a perspective
        // projection is view depth -- exactly what picks a cascade, with no extra uniform needed.
        float shadowLit = ShadowLit(input.worldPos, shadowN, input.position.w);
        texel.rgb *= lerp(1.0 - shadow_params.w, 1.0, shadowLit);
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
