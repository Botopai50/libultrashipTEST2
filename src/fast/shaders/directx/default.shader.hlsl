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
@if(o_toon || o_water)
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
    float3 toon_camera_pos;
    float toon_rim_enabled;
    float toon_rim_intensity;
    float toon_rim_width;
    float toon_rim_softness;
    float toon_rim_direction_influence;
}
@end

@if(o_water)
Texture2D<float4> water_scene_color : register(t6);
Texture2D<float> water_scene_depth : register(t7);
#if WATER_DYNAMIC_MSAA
Texture2DMS<float> water_scene_depth_ms : register(t8);
#endif
SamplerState water_scene_sampler : register(s6);

cbuffer PerWaterCB : register(b3) {
    float4 water_shallow_color;
    float4 water_deep_color;
    float4 water_foam_color;
    float4 water_caustic_color;
    float3 water_camera_pos;
    float water_fade_distance;
    float3 water_light_dir;
    float water_foam_thickness;
    float3 water_light_color;
    float water_normal_scale;
    float2 water_uv_speed1;
    float2 water_uv_speed2;
    float water_normal_strength;
    float water_reflection_intensity;
    float water_reflection_distortion;
    float water_fresnel_power;
    float water_specular_threshold;
    float water_specular_intensity;
    float water_caustic_scale;
    float water_caustic_strength;
    float water_caustic_thickness;
    float water_near_plane;
    float water_far_plane;
    float _water_material_pad;
    float2 water_viewport_size;
    float water_depth_available;
    float water_msaa_samples;
    float water_time_seconds;
    float3 _water_pad;
}

float2 waterHash22(float2 p) {
    float3 p3 = frac(float3(p.x, p.y, p.x) * float3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.xx + p3.yz) * p3.zy);
}

float waterCausticWeb(float2 p, float thickness) {
    float2 cell = floor(p);
    float2 local = frac(p);
    float nearest = 8.0;
    float secondNearest = 8.0;
    [unroll] for (int y = -1; y <= 1; ++y) {
        [unroll] for (int x = -1; x <= 1; ++x) {
            float2 neighbor = float2((float)x, (float)y);
            float2 delta = neighbor + waterHash22(cell + neighbor) - local;
            float distanceSquared = dot(delta, delta);
            if (distanceSquared < nearest) {
                secondNearest = nearest;
                nearest = distanceSquared;
            } else {
                secondNearest = min(secondNearest, distanceSquared);
            }
        }
    }
    float edgeDistance = max(sqrt(secondNearest) - sqrt(nearest), 0.0);
    float edgeWidth = clamp(thickness, 0.005, 0.3);
    float antialias = max(fwidth(edgeDistance) * 0.75, 0.002);
    return 1.0 - smoothstep(max(edgeWidth - antialias, 0.0), edgeWidth + antialias, edgeDistance);
}

float waterLinearDepth(float depth01) {
    return (water_near_plane * water_far_plane) /
           max(water_far_plane - depth01 * (water_far_plane - water_near_plane), 0.0001);
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
@if(o_toon || o_water)
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
    @if(o_toon || o_water)
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

    @if(o_shadow_solid)
        // Screen-space antialiasing reconstructs a clean solid contour from the fixed 96x96 coverage mask.
        float shadowEdgeWidth = max(fwidth(texVal0.a) * 0.75, 1.0 / 255.0);
        texVal0.a = smoothstep(0.5 - shadowEdgeWidth, 0.5 + shadowEdgeWidth, texVal0.a);
        if (texVal0.a <= 1.0 / 255.0) discard;
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

    @if(o_water)
        float waterTime = water_time_seconds;
        float2 waterBaseUv = input.worldPos.xz * max(water_normal_scale, 0.00001);
        float2 waterUv1 = waterBaseUv + water_uv_speed1 * waterTime;
        float2 waterUv2 = float2(-waterBaseUv.y, waterBaseUv.x) + water_uv_speed2 * waterTime;

        float2 waterSlope1;
        @if(o_textures[0])
            uint waterWidth1, waterHeight1;
            g_texture0.GetDimensions(waterWidth1, waterHeight1);
            float2 waterTexel1 = 1.0 / max(float2(waterWidth1, waterHeight1), float2(1.0, 1.0));
            float waterH1 = dot(g_texture0.Sample(g_sampler0, waterUv1).rgb, float3(0.299, 0.587, 0.114));
            waterSlope1 = float2(
                dot(g_texture0.Sample(g_sampler0, waterUv1 + float2(waterTexel1.x, 0.0)).rgb,
                    float3(0.299, 0.587, 0.114)) - waterH1,
                dot(g_texture0.Sample(g_sampler0, waterUv1 + float2(0.0, waterTexel1.y)).rgb,
                    float3(0.299, 0.587, 0.114)) - waterH1);
        @else
            waterSlope1 = float2(cos(waterUv1.x * 6.283), sin(waterUv1.y * 6.283)) * 0.08;
        @end

        float2 waterSlope2;
        @if(o_textures[1])
            uint waterWidth2, waterHeight2;
            g_texture1.GetDimensions(waterWidth2, waterHeight2);
            float2 waterTexel2 = 1.0 / max(float2(waterWidth2, waterHeight2), float2(1.0, 1.0));
            float waterH2 = dot(g_texture1.Sample(g_sampler1, waterUv2).rgb, float3(0.299, 0.587, 0.114));
            waterSlope2 = float2(
                dot(g_texture1.Sample(g_sampler1, waterUv2 + float2(waterTexel2.x, 0.0)).rgb,
                    float3(0.299, 0.587, 0.114)) - waterH2,
                dot(g_texture1.Sample(g_sampler1, waterUv2 + float2(0.0, waterTexel2.y)).rgb,
                    float3(0.299, 0.587, 0.114)) - waterH2);
        @elseif(o_textures[0])
            uint waterWidth2, waterHeight2;
            g_texture0.GetDimensions(waterWidth2, waterHeight2);
            float2 waterTexel2 = 1.0 / max(float2(waterWidth2, waterHeight2), float2(1.0, 1.0));
            float waterH2 = dot(g_texture0.Sample(g_sampler0, waterUv2).rgb, float3(0.299, 0.587, 0.114));
            waterSlope2 = float2(
                dot(g_texture0.Sample(g_sampler0, waterUv2 + float2(waterTexel2.x, 0.0)).rgb,
                    float3(0.299, 0.587, 0.114)) - waterH2,
                dot(g_texture0.Sample(g_sampler0, waterUv2 + float2(0.0, waterTexel2.y)).rgb,
                    float3(0.299, 0.587, 0.114)) - waterH2);
        @else
            waterSlope2 = float2(sin(waterUv2.x * 5.17), cos(waterUv2.y * 7.11)) * 0.08;
        @end

        float3 waterDx = ddx(input.worldPos);
        float3 waterDy = ddy(input.worldPos);
        float3 waterGeometricN = normalize(cross(waterDx, waterDy));
        float3 waterV = normalize(water_camera_pos - input.worldPos);
        if (dot(waterGeometricN, waterV) < 0.0) waterGeometricN = -waterGeometricN;
        float3 waterTangent = normalize(waterDx);
        float3 waterBitangent = normalize(cross(waterGeometricN, waterTangent));
        float2 waterSlope = (waterSlope1 + waterSlope2) * max(water_normal_strength, 0.0) * 6.0;
        float3 waterN = normalize(waterGeometricN - waterTangent * waterSlope.x - waterBitangent * waterSlope.y);

        float2 waterScreenUv = screenSpace.xy / max(water_viewport_size, float2(1.0, 1.0));
        float waterDepthGap = max(water_fade_distance, 1.0);
        if (water_depth_available > 0.5) {
            int2 waterPixel = int2(clamp(screenSpace.xy, float2(0.0, 0.0), water_viewport_size - 1.0));
#if WATER_DYNAMIC_MSAA
            float waterSceneZ = water_msaa_samples > 1.5
                                    ? water_scene_depth_ms.Load(waterPixel, 0)
                                    : water_scene_depth.Load(int3(waterPixel, 0));
#else
            float waterSceneZ = water_scene_depth.Load(int3(waterPixel, 0));
#endif
            waterDepthGap = max(waterLinearDepth(waterSceneZ) - waterLinearDepth(screenSpace.z), 0.0);
        }
        float waterDeepFactor = smoothstep(0.0, max(water_fade_distance, 1.0), waterDepthGap);
        float4 waterDepthTint = lerp(water_shallow_color, water_deep_color, waterDeepFactor);
        float4 waterBaseColor = texel;
        waterBaseColor.rgb *= waterDepthTint.rgb;
        waterBaseColor.a = waterDepthTint.a;

        float2 waterCausticUv = input.worldPos.xz * max(water_caustic_scale, 0.00001);
        float2 waterCausticWarp = float2(
            sin(waterCausticUv.y * 1.71 + waterTime * 0.73) +
                sin(waterCausticUv.x * 0.67 - waterTime * 0.41),
            cos(waterCausticUv.x * 1.43 - waterTime * 0.61) +
                cos(waterCausticUv.y * 0.79 + waterTime * 0.53)) * 0.16;
        waterCausticUv += waterCausticWarp + water_uv_speed1 * waterTime * 8.0;
        float waterCaustic = waterCausticWeb(waterCausticUv, water_caustic_thickness);

        float waterFresnel = pow(1.0 - saturate(dot(waterN, waterV)), max(water_fresnel_power, 0.01));
        float2 waterMirrorUv = float2(waterScreenUv.x, 1.0 - waterScreenUv.y);
        float2 waterReflectUv = clamp(waterMirrorUv + float2(waterN.x, -waterN.z) * water_reflection_distortion,
                                      float2(0.002, 0.002), float2(0.998, 0.998));
        float3 waterSceneReflection = water_scene_color.Sample(water_scene_sampler, waterReflectUv).rgb;
        float waterReflectionAmount = saturate(lerp(0.20, 1.0, waterFresnel) * water_reflection_intensity);
        waterBaseColor.rgb = lerp(waterBaseColor.rgb, waterSceneReflection,
                                  waterReflectionAmount);

        float waterCausticFacing = lerp(1.0, 0.65, waterFresnel);
        float waterCausticAmount = saturate(waterCaustic * water_caustic_strength *
                                            water_caustic_color.a * waterCausticFacing);
        waterBaseColor = lerp(waterBaseColor, water_caustic_color, waterCausticAmount);

        float3 waterL = normalize(water_light_dir);
        float3 waterH = normalize(waterL + waterV);
        float waterSpecular = step(saturate(water_specular_threshold), max(dot(waterN, waterH), 0.0));
        waterSpecular *= step(0.0, dot(waterN, waterL)) * max(water_specular_intensity, 0.0);
        waterBaseColor.rgb += water_light_color * waterSpecular;

        float waterFoam = 0.0;
        if (water_depth_available > 0.5) {
            float waterContact = 1.0 - smoothstep(0.0, max(water_foam_thickness, 0.001), waterDepthGap);
            float waterIrregular = 1.0 - waterCaustic;
            waterFoam = saturate(waterContact * lerp(0.45, 1.0, waterIrregular));
        }
        waterBaseColor = lerp(waterBaseColor, water_foam_color, waterFoam * water_foam_color.a);
        texel = saturate(waterBaseColor);
    @end

    // SOH [Enhancement] Toon lighting: re-light the (white-shaded) albedo with the single dominant
    // light through a soft half-Lambert ramp.
    @if(o_toon)
        float3 albedoColor = texel.rgb;
        float3 toonN = normalize(input.normal);
        float3 toonL = normalize(toon_light_dir);
        float toonNL = dot(toonN, toonL) * 0.5 + 0.5;
        float toonRamp = smoothstep(toon_ramp_center - toon_ramp_softness, toon_ramp_center + toon_ramp_softness, toonNL);
        float3 toonLit = toon_ambient + toon_light_color * toon_highlight_intensity;
        float3 toonShadow = lerp(toonLit, toon_ambient, toon_shadow_intensity);
        if (toon_debug > 0.5) {
            // Diagnostic view: flat white on the lit side of the ramp, flat black in shadow, albedo
            // discarded — makes it obvious which draws are receiving toon lighting.
            texel.rgb = float3(toonRamp, toonRamp, toonRamp);
        } else {
            texel.rgb = clamp(texel.rgb * lerp(toonShadow, toonLit, toonRamp), 0.0, 1.0);
            // Same low-cost, directional inner silhouette band as the OpenGL path. The toon shader
            // variant identifies lit actor materials; !o_alpha excludes translucent/particle draws.
            @if(!o_alpha)
                if (toon_rim_enabled > 0.5) {
                    float3 viewDelta = toon_camera_pos - input.worldPos;
                    float3 V = viewDelta * rsqrt(max(dot(viewDelta, viewDelta), 0.000001));
                    float facing = saturate(dot(toonN, V));
                    float facingGradient = max(fwidth(facing), 0.0001);
                    float silhouetteDistancePixels = facing / facingGradient;
                    float widthControl = saturate(toon_rim_width);
                    float rimWidthPixels = lerp(0.25, 1.0, widthControl * widthControl);
                    float smoothControl = saturate(toon_rim_softness / 0.15);
                    float featherPixels = lerp(0.30, 0.65, smoothControl);
                    float rimBand = 1.0 - smoothstep(rimWidthPixels - featherPixels,
                                                     rimWidthPixels + featherPixels,
                                                     silhouetteDistancePixels);

                    float lightSide = smoothstep(-0.35, 0.10, dot(toonN, toonL));
                    float backLighting = smoothstep(0.0, 0.65, dot(-V, toonL));
                    float directionalMask = lightSide * lerp(0.25, 1.0, backLighting);
                    float directionInfluence = clamp(toon_rim_direction_influence, 0.25, 1.0);
                    directionalMask = lerp(1.0, directionalMask, directionInfluence);

                    float materialRimMask = 1.0;
                    float rimMask = saturate(rimBand * directionalMask * materialRimMask *
                                             max(toon_rim_intensity, 0.0));
                    float3 rimColor = lerp(float3(1.0, 1.0, 1.0), saturate(toon_light_color), 0.35);
                    texel.rgb = lerp(texel.rgb, rimColor, rimMask);
                }
            @end
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
