@prism(type='metal', name='Fast3D Metal Shader', version='1.0.0', description='Ported shader to prism', author='Emill & Prism Team')

#include <metal_stdlib>
using namespace metal;

// BEGIN VERTEX SHADER
struct FrameUniforms {
    int frameCount;
    float noiseScale;
};

struct DrawUniforms {
    // SOH [Enhancement] Sized to 6 (was 2) to match the C++ DrawUniforms (textureFiltering[SHADER_MAX_TEXTURES]),
    // so the toon fields below land at the same byte offset in both structs.
    int textureFiltering[6];
    // SOH [Enhancement] Toon lighting (unconditional so the CB layout matches the C++ struct for
    // every shader variant). packed_float3 is 12 bytes / 4-aligned, matching simd::float1[3].
    packed_float3 toonLightDir;
    packed_float3 toonLightColor;
    packed_float3 toonAmbient;
    float toonRampCenter;
    float toonRampSoftness;
    float toonHighlightIntensity;
    float toonShadowIntensity;
    float toonDebug;
    float4 waterShallowColor;
    float4 waterDeepColor;
    float4 waterFoamColor;
    float4 waterCausticColor;
    packed_float3 waterCameraPos;
    float waterFadeDistance;
    packed_float3 waterLightDir;
    float waterFoamThickness;
    packed_float3 waterLightColor;
    float waterNormalScale;
    float2 waterUvSpeed1;
    float2 waterUvSpeed2;
    float waterNormalStrength;
    float waterReflectionIntensity;
    float waterReflectionDistortion;
    float waterFresnelPower;
    float waterSpecularThreshold;
    float waterSpecularIntensity;
    float waterCausticScale;
    float waterCausticStrength;
    float waterCausticThickness;
    float waterNearPlane;
    float waterFarPlane;
    float waterMaterialPadding;
    float waterTimeSeconds;
    packed_float3 waterPadding;
};

struct Vertex {
    float4 position [[attribute(@{get_vertex_index()})]];
    @{update_floats(4)}
    @for(i in 0..2)
        @if(o_textures[i])
            float2 texCoord@{i} [[attribute(@{get_vertex_index()})]];
            @{update_floats(2)}
            @for(j in 0..2)
                @if(o_clamp[i][j])
                    @if(j == 0)
                        float texClampS@{i} [[attribute(@{get_vertex_index()})]];
                    @else
                        float texClampT@{i} [[attribute(@{get_vertex_index()})]];
                    @end
                    @{update_floats(1)}
                @end
            @end
        @end
    @end
    @if(o_fog)
        float4 fog [[attribute(@{get_vertex_index()})]];
        @{update_floats(4)}
    @end
    @if(o_grayscale)
        float4 grayscale [[attribute(@{get_vertex_index()})]];
        @{update_floats(4)}
    @end
    @if(o_toon)
        float3 normal [[attribute(@{get_vertex_index()})]];
        @{update_floats(3)}
    @end
    @if(o_water)
        float3 worldPos [[attribute(@{get_vertex_index()})]];
        @{update_floats(3)}
    @end
    @for(i in 0..o_inputs)
        @if(o_alpha)
            float4 input@{i + 1} [[attribute(@{get_vertex_index()})]];
            @{update_floats(4)}
        @else
            float3 input@{i + 1} [[attribute(@{get_vertex_index()})]];
            @{update_floats(3)}
        @end
    @end
};

struct ProjectedVertex {
    @for(i in 0..2)
        @if(o_textures[i])
            float2 texCoord@{i};
            @for(j in 0..2)
                @if(o_clamp[i][j])
                    @if(j == 0)
                        float texClampS@{i};
                    @else
                        float texClampT@{i};
                    @end
                @end
            @end
        @end
    @end
    @if(o_fog)
        float4 fog;
    @end
    @if(o_grayscale)
        float4 grayscale;
    @end
    @if(o_toon)
        float3 normal;
    @end
    @if(o_water)
        float3 worldPos;
    @end
    @for(i in 0..o_inputs)
        @if(o_alpha)
            float4 input@{i + 1};
        @else
            float3 input@{i + 1};
        @end
    @end
    float4 position [[position]];
};

vertex ProjectedVertex vertexShader(Vertex in [[stage_in]]) {
    ProjectedVertex out;
    @for(i in 0..2)
        @if(o_textures[i])
            out.texCoord@{i} = in.texCoord@{i};
            @for(j in 0..2)
                @if(o_clamp[i][j])
                    @if(j == 0)
                        out.texClampS@{i} = in.texClampS@{i};
                    @else
                        out.texClampT@{i} = in.texClampT@{i};
                    @end
                @end
            @end
        @end
    @end
    @if(o_fog)
        out.fog = in.fog;
    @end
    @if(o_grayscale)
        out.grayscale = in.grayscale;
    @end
    @if(o_toon)
        out.normal = in.normal;
    @end
    @if(o_water)
        out.worldPos = in.worldPos;
    @end
    @for(i in 0..o_inputs)
         out.input@{i + 1} = in.input@{i + 1};
    @end
    out.position = in.position;
    return out;
}
// END - BEGIN FRAGMENT SHADER

float mod(float x, float y) {
    return float(x - y * floor(x / y));
}

float3 mod(float3 a, float3 b) {
    return float3(a.x - b.x * floor(a.x / b.x), a.y - b.y * floor(a.y / b.y), a.z - b.z * floor(a.z / b.z));
}

float4 mod(float4 a, float4 b) {
    return float4(a.x - b.x * floor(a.x / b.x), a.y - b.y * floor(a.y / b.y), a.z - b.z * floor(a.z / b.z), a.w - b.w * floor(a.w / b.w));
}

#define WRAP(x, low, high) mod((x)-(low), (high)-(low)) + (low)
#define TEX_OFFSET(tex, texSmplr, texCoord, off, texSize) tex.sample(texSmplr, texCoord - off / texSize)

float4 filter3point(thread const texture2d<float> tex, thread const sampler texSmplr, thread const float2& texCoord, thread const float2& texSize) {
    float2 offset = fract((texCoord * texSize) - float2(0.5));
    offset -= float2(step(1.0, offset.x + offset.y));
    float4 c0 = TEX_OFFSET(tex, texSmplr, texCoord, offset, texSize);
    float4 c1 = TEX_OFFSET(tex, texSmplr, texCoord, float2(offset.x - sign(offset.x), offset.y), texSize);
    float4 c2 = TEX_OFFSET(tex, texSmplr, texCoord, float2(offset.x, offset.y - sign(offset.y)), texSize);
    return c0 + abs(offset.x) * (c1 - c0) + abs(offset.y) * (c2 - c0);
}

float4 hookTexture2D(thread const texture2d<float> tex, thread const sampler texSmplr, thread const float2& uv, thread const float2& texSize, thread const int filtering) {
@if(o_three_point_filtering)
    if(filtering == @{FILTER_THREE_POINT}) {
        return filter3point(tex, texSmplr, uv, texSize);
    }
@end
    return tex.sample(texSmplr, uv);
}

float random(float3 value) {
    float random = dot(sin(value), float3(12.9898, 78.233, 37.719));
    return fract(sin(random) * 143758.5453);
}

float2 waterHash22(float2 p) {
    float3 p3 = fract(float3(p.x, p.y, p.x) * float3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

float waterCausticWeb(float2 p, float thickness) {
    float2 cell = floor(p);
    float2 local = fract(p);
    float nearest = 8.0;
    float secondNearest = 8.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float2 neighbor = float2(float(x), float(y));
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
    float antialias = max((abs(dfdx(edgeDistance)) + abs(dfdy(edgeDistance))) * 0.75, 0.002);
    return 1.0 - smoothstep(max(edgeWidth - antialias, 0.0), edgeWidth + antialias, edgeDistance);
}

fragment float4 fragmentShader(
    ProjectedVertex in [[stage_in]],
    constant FrameUniforms &frameUniforms [[buffer(0)]],
    constant DrawUniforms &drawUniforms [[buffer(1)]]
@if(o_textures[0])
    , texture2d<float> uTex0 [[texture(0)]], sampler uTex0Smplr [[sampler(0)]]
@end
@if(o_textures[1])
    , texture2d<float> uTex1 [[texture(1)]], sampler uTex1Smplr [[sampler(1)]]
@end
@if(o_masks[0])
    , texture2d<float> uTexMask0 [[texture(2)]]
@end
@if(o_masks[1])
    , texture2d<float> uTexMask1 [[texture(3)]]
@end
@if(o_blend[0])
    , texture2d<float> uTexBlend0 [[texture(4)]]
@end
@if(o_blend[1])
    , texture2d<float> uTexBlend1 [[texture(5)]]
@end
) {
    @for(i in 0..2)
        @if(o_textures[i])
            @{s = o_clamp[i][0]}
            @{t = o_clamp[i][1]}
            float2 texSize@{i} = float2(uTex@{i}.get_width(), uTex@{i}.get_height());
            @if(!s && !t)
                float2 vTexCoordAdj@{i} = in.texCoord@{i};
            @else
                @if(s && t)
                    float2 vTexCoordAdj@{i} = fast::clamp(in.texCoord@{i}, float2(0.5) / texSize@{i}, float2(in.texClampS@{i}, in.texClampT@{i}));
                @elseif(s)
                    float2 vTexCoordAdj@{i} = float2(fast::clamp(in.texCoord@{i}.x, 0.5 / texSize@{i}.x, in.texClampS@{i}), in.texCoord@{i}.y);
                @else
                    float2 vTexCoordAdj@{i} = float2(in.texCoord@{i}.x, fast::clamp(in.texCoord@{i}.y, 0.5 / texSize@{i}.y, in.texClampT@{i}));
                @end
            @end

            float4 texVal@{i} = hookTexture2D(uTex@{i}, uTex@{i}Smplr, vTexCoordAdj@{i}, texSize@{i}, drawUniforms.textureFiltering[@{i}]);

            @if(o_masks[i])
                float2 maskSize@{i} = float2(uTexMask@{i}.get_width(), uTexMask@{i}.get_height());
                float4 maskVal@{i} = hookTexture2D(uTexMask@{i}, uTex@{i}Smplr, vTexCoordAdj@{i}, maskSize@{i}, drawUniforms.textureFiltering[@{i}]);
                @if(o_blend[i])
                    float4 blendVal@{i} = hookTexture2D(uTexBlend@{i}, uTex@{i}Smplr, vTexCoordAdj@{i}, texSize@{i}, drawUniforms.textureFiltering[@{i}]);
                @else
                    float4 blendVal@{i} = float4(0, 0, 0, 0);
                @end

                texVal@{i} = mix(texVal@{i}, blendVal@{i}, maskVal@{i}.w);
            @end
        @end
    @end
    
    @if(o_shadow_solid)
        // Screen-space antialiasing reconstructs a clean solid contour from the fixed 96x96 coverage mask.
        float shadowEdgeWidth = max(fwidth(texVal0.w) * 0.75, 1.0 / 255.0);
        texVal0.w = smoothstep(0.5 - shadowEdgeWidth, 0.5 + shadowEdgeWidth, texVal0.w);
        if (texVal0.w <= 1.0 / 255.0) discard_fragment();
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
                    texel.w = WRAP(texel.w, -1.01, 1.01);
                @else
                    texel.w = WRAP(texel.w, -0.51, 1.51);
                @end
            @end

            @if(o_c[c][0][2] == SHADER_COMBINED)
                texel.xyz = WRAP(texel.xyz, -1.01, 1.01);
            @else
                texel.xyz = WRAP(texel.xyz, -0.51, 1.51);
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
        if (texel.w > 0.19) texel.w = 1.0; else discard_fragment();
    @end

    texel = WRAP(texel, -0.51, 1.51);
    texel = clamp(texel, 0.0, 1.0);
    // TODO discard if alpha is 0?

    @if(o_water)
        float2 waterBaseUv = in.worldPos.xz * max(drawUniforms.waterNormalScale, 0.00001);
        float2 waterUv1 = waterBaseUv + float2(drawUniforms.waterUvSpeed1) * drawUniforms.waterTimeSeconds;
        float2 waterUv2 = float2(-waterBaseUv.y, waterBaseUv.x) +
                          float2(drawUniforms.waterUvSpeed2) * drawUniforms.waterTimeSeconds;
        float2 waterSlope1;
        @if(o_textures[0])
            float2 waterTexel1 = 1.0 / float2(uTex0.get_width(), uTex0.get_height());
            float waterH1 = dot(uTex0.sample(uTex0Smplr, waterUv1).xyz, float3(0.299, 0.587, 0.114));
            waterSlope1 = float2(
                dot(uTex0.sample(uTex0Smplr, waterUv1 + float2(waterTexel1.x, 0.0)).xyz,
                    float3(0.299, 0.587, 0.114)) - waterH1,
                dot(uTex0.sample(uTex0Smplr, waterUv1 + float2(0.0, waterTexel1.y)).xyz,
                    float3(0.299, 0.587, 0.114)) - waterH1);
        @else
            waterSlope1 = float2(cos(waterUv1.x * 6.283), sin(waterUv1.y * 6.283)) * 0.08;
        @end
        float2 waterSlope2;
        @if(o_textures[1])
            float2 waterTexel2 = 1.0 / float2(uTex1.get_width(), uTex1.get_height());
            float waterH2 = dot(uTex1.sample(uTex1Smplr, waterUv2).xyz, float3(0.299, 0.587, 0.114));
            waterSlope2 = float2(
                dot(uTex1.sample(uTex1Smplr, waterUv2 + float2(waterTexel2.x, 0.0)).xyz,
                    float3(0.299, 0.587, 0.114)) - waterH2,
                dot(uTex1.sample(uTex1Smplr, waterUv2 + float2(0.0, waterTexel2.y)).xyz,
                    float3(0.299, 0.587, 0.114)) - waterH2);
        @else
            waterSlope2 = float2(sin(waterUv2.x * 5.17), cos(waterUv2.y * 7.11)) * 0.08;
        @end
        float3 waterDx = dfdx(in.worldPos);
        float3 waterDy = dfdy(in.worldPos);
        float3 waterGeometricN = normalize(cross(waterDx, waterDy));
        float3 waterV = normalize(float3(drawUniforms.waterCameraPos) - in.worldPos);
        if (dot(waterGeometricN, waterV) < 0.0) waterGeometricN = -waterGeometricN;
        float3 waterTangent = normalize(waterDx);
        float3 waterBitangent = normalize(cross(waterGeometricN, waterTangent));
        float2 waterSlope = (waterSlope1 + waterSlope2) * max(drawUniforms.waterNormalStrength, 0.0) * 6.0;
        float3 waterN = normalize(waterGeometricN - waterTangent * waterSlope.x - waterBitangent * waterSlope.y);

        float legacyDepthHint = clamp(texel.w, 0.0, 1.0);
        float4 waterDepthTint = mix(drawUniforms.waterShallowColor, drawUniforms.waterDeepColor,
                                    legacyDepthHint);
        float4 waterBaseColor = texel;
        waterBaseColor.xyz *= waterDepthTint.xyz;
        waterBaseColor.w = waterDepthTint.w;

        float2 waterCausticUv = in.worldPos.xz * max(drawUniforms.waterCausticScale, 0.00001);
        float2 waterCausticWarp = float2(
            sin(waterCausticUv.y * 1.71 + drawUniforms.waterTimeSeconds * 0.73) +
                sin(waterCausticUv.x * 0.67 - drawUniforms.waterTimeSeconds * 0.41),
            cos(waterCausticUv.x * 1.43 - drawUniforms.waterTimeSeconds * 0.61) +
                cos(waterCausticUv.y * 0.79 + drawUniforms.waterTimeSeconds * 0.53)) * 0.16;
        waterCausticUv += waterCausticWarp +
                           float2(drawUniforms.waterUvSpeed1) * drawUniforms.waterTimeSeconds * 8.0;
        float waterCaustic = waterCausticWeb(waterCausticUv, drawUniforms.waterCausticThickness);
        float waterFresnel = pow(1.0 - clamp(dot(waterN, waterV), 0.0, 1.0),
                                 max(drawUniforms.waterFresnelPower, 0.01));
        float3 waterSkyTint = mix(float3(drawUniforms.waterDeepColor.xyz),
                                  float3(drawUniforms.waterLightColor), 0.35);
        float waterReflectionAmount = clamp(mix(0.20, 1.0, waterFresnel) *
                                            drawUniforms.waterReflectionIntensity, 0.0, 1.0);
        waterBaseColor.xyz = mix(waterBaseColor.xyz, waterSkyTint, waterReflectionAmount);
        float waterCausticFacing = mix(1.0, 0.65, waterFresnel);
        float waterCausticAmount = clamp(waterCaustic * drawUniforms.waterCausticStrength *
                                         drawUniforms.waterCausticColor.w * waterCausticFacing, 0.0, 1.0);
        waterBaseColor = mix(waterBaseColor, drawUniforms.waterCausticColor, waterCausticAmount);
        float3 waterL = normalize(float3(drawUniforms.waterLightDir));
        float3 waterH = normalize(waterL + waterV);
        float waterSpecular = step(clamp(drawUniforms.waterSpecularThreshold, 0.0, 1.0),
                                   max(dot(waterN, waterH), 0.0));
        waterSpecular *= step(0.0, dot(waterN, waterL)) * max(drawUniforms.waterSpecularIntensity, 0.0);
        waterBaseColor.xyz += float3(drawUniforms.waterLightColor) * waterSpecular;
        texel = clamp(waterBaseColor, 0.0, 1.0);
    @end

    // SOH [Enhancement] Toon lighting: re-light the (white-shaded) albedo with the single dominant
    // light through a soft half-Lambert ramp.
    @if(o_toon)
        float3 toonN = normalize(in.normal);
        float3 toonL = normalize(float3(drawUniforms.toonLightDir));
        float toonNL = dot(toonN, toonL) * 0.5 + 0.5;
        float toonRamp = smoothstep(drawUniforms.toonRampCenter - drawUniforms.toonRampSoftness,
                                    drawUniforms.toonRampCenter + drawUniforms.toonRampSoftness, toonNL);
        float3 toonLit = float3(drawUniforms.toonAmbient) +
                         float3(drawUniforms.toonLightColor) * drawUniforms.toonHighlightIntensity;
        float3 toonShadow = mix(toonLit, float3(drawUniforms.toonAmbient), drawUniforms.toonShadowIntensity);
        if (drawUniforms.toonDebug > 0.5) {
            // Diagnostic view: flat white on the lit side of the ramp, flat black in shadow, albedo
            // discarded — makes it obvious which draws are receiving toon lighting.
            texel.xyz = float3(toonRamp);
        } else {
            texel.xyz = clamp(texel.xyz * mix(toonShadow, toonLit, toonRamp), 0.0, 1.0);
        }
    @end

    @if(o_fog)
        @if(o_alpha)
            texel = float4(mix(texel.xyz, in.fog.xyz, in.fog.w), texel.w);
        @else
            texel = mix(texel, in.fog.xyz, in.fog.w);
        @end
    @end

    @if(o_grayscale)
        float intensity = (texel.x + texel.y + texel.z) / 3.0;
        float3 new_texel = in.grayscale.xyz * intensity;
        texel.xyz = mix(texel.xyz, new_texel, in.grayscale.w);
    @end

    @if(o_alpha && o_noise)
        float2 coords = in.position.xy * frameUniforms.noiseScale;
        texel.w *= round(saturate(random(float3(floor(coords), float(frameUniforms.frameCount))) + texel.w - 0.5));
    @end

    @if(o_alpha)
        @if(o_alpha_threshold)
            if (texel.w < 8.0 / 256.0) discard_fragment();
        @end
        @if(o_invisible)
            texel.w = 0.0;
        @end
        return texel;
    @else
        return float4(texel, 1.0);
    @end
}
