@prism(type='fragment', name='Fast3D Fragment Shader', version='1.0.0', description='Ported shader to prism', author='Emill & Prism Team')

@{GLSL_VERSION}

@if(opengles && o_water)
precision highp float;
@end

@if(core_opengl || opengles)
out vec4 vOutColor;
@end

@for(i in 0..2)
    @if(o_textures[i])
        @{attr} vec2 vTexCoord@{i};
        @for(j in 0..2)
            @if(o_clamp[i][j])
                @if(j == 0)
                    @{attr} float vTexClampS@{i};
                @else
                    @{attr} float vTexClampT@{i};
                @end
            @end
        @end
    @end
@end

@if(o_fog) @{attr} vec4 vFog;
@if(o_grayscale) @{attr} vec4 vGrayscaleColor;
@if(o_toon) @{attr} vec3 vNormal;
@if(o_toon || o_water) @{attr} vec3 vWorldPos;

@for(i in 0..o_inputs)
    @if(o_alpha)
        @{attr} vec4 vInput@{i + 1};
    @else
        @{attr} vec3 vInput@{i + 1};
    @end
@end

@if(o_textures[0]) uniform sampler2D uTex0;
@if(o_textures[1]) uniform sampler2D uTex1;

@if(o_masks[0]) uniform sampler2D uTexMask0;
@if(o_masks[1]) uniform sampler2D uTexMask1;

@if(o_blend[0]) uniform sampler2D uTexBlend0;
@if(o_blend[1]) uniform sampler2D uTexBlend1;

uniform int frame_count;
uniform float noise_scale;

// SOH [Enhancement] Toon lighting (single dominant light + soft ramp).
@if(o_toon)
uniform vec3 toon_light_dir;
uniform vec3 toon_light_color;
uniform vec3 toon_ambient;
uniform float toon_ramp_center;
uniform float toon_ramp_softness;
uniform float toon_highlight_intensity;
uniform float toon_shadow_intensity;
uniform float toon_debug;
uniform vec3 toon_camera_pos;
uniform float toon_rim_enabled;
uniform float toon_rim_intensity;
uniform float toon_rim_width;
uniform float toon_rim_softness;
uniform float toon_rim_direction_influence;
@end

@if(o_water)
uniform sampler2D water_scene_color;
uniform sampler2D water_scene_depth;
uniform vec4 water_shallow_color;
uniform vec4 water_deep_color;
uniform vec4 water_foam_color;
uniform vec4 water_caustic_color;
uniform vec3 water_camera_pos;
uniform vec3 water_light_dir;
uniform vec3 water_light_color;
uniform vec2 water_uv_speed1;
uniform vec2 water_uv_speed2;
uniform float water_fade_distance;
uniform float water_foam_thickness;
uniform float water_normal_scale;
uniform float water_normal_strength;
uniform float water_reflection_intensity;
uniform float water_reflection_distortion;
uniform float water_fresnel_power;
uniform float water_specular_threshold;
uniform float water_specular_intensity;
uniform float water_caustic_scale;
uniform float water_caustic_strength;
uniform float water_caustic_thickness;
uniform float water_near_plane;
uniform float water_far_plane;
uniform vec2 water_viewport_size;
uniform float water_depth_available;
uniform float water_time_seconds;

vec2 waterHash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

// One antialiased Worley edge field supplies both the bright surface web and shoreline breakup.
// Reusing it keeps the reference-style caustics to one lightweight 3x3 search and no extra texture pass.
float waterCausticWeb(vec2 p, float thickness) {
    vec2 cell = floor(p);
    vec2 local = fract(p);
    float nearest = 8.0;
    float secondNearest = 8.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 neighbor = vec2(float(x), float(y));
            vec2 delta = neighbor + waterHash22(cell + neighbor) - local;
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
    float zNdc = depth01 * 2.0 - 1.0;
    @if(opengles)
        // The GLES vertex path compresses clip-space Z to preserve the port's legacy depth behavior.
        // Undo that mapping before applying the ordinary perspective-depth inverse.
        zNdc /= 0.3;
    @end
    return (2.0 * water_near_plane * water_far_plane) /
           max(water_far_plane + water_near_plane - zNdc * (water_far_plane - water_near_plane), 0.0001);
}
@end

uniform int texture_width[2];
uniform int texture_height[2];
uniform int texture_filtering[2];

#define TEX_OFFSET(off) @{texture}(tex, texCoord - off / texSize)
#define WRAP(x, low, high) mod((x)-(low), (high)-(low)) + (low)

float random(in vec3 value) {
    float random = dot(sin(value), vec3(12.9898, 78.233, 37.719));
    return fract(sin(random) * 143758.5453);
}

vec4 fromLinear(vec4 linearRGB){
    bvec3 cutoff = lessThan(linearRGB.rgb, vec3(0.0031308));
    vec3 higher = vec3(1.055)*pow(linearRGB.rgb, vec3(1.0/2.4)) - vec3(0.055);
    vec3 lower = linearRGB.rgb * vec3(12.92);
    return vec4(mix(higher, lower, cutoff), linearRGB.a);
}

vec4 filter3point(in sampler2D tex, in vec2 texCoord, in vec2 texSize) {
    vec2 offset = fract(texCoord*texSize - vec2(0.5));
    offset -= step(1.0, offset.x + offset.y);
    vec4 c0 = TEX_OFFSET(offset);
    vec4 c1 = TEX_OFFSET(vec2(offset.x - sign(offset.x), offset.y));
    vec4 c2 = TEX_OFFSET(vec2(offset.x, offset.y - sign(offset.y)));
    return c0 + abs(offset.x)*(c1-c0) + abs(offset.y)*(c2-c0);
}

vec4 hookTexture2D(in int id, sampler2D tex, in vec2 uv, in vec2 texSize) {
@if(o_three_point_filtering)
    if(texture_filtering[id] == @{FILTER_THREE_POINT}) {
        return filter3point(tex, uv, texSize);
    }
@end
    return @{texture}(tex, uv);
}

#define TEX_SIZE(tex) vec2(texture_width[tex], texture_height[tex])

void main() {
    @for(i in 0..2)
        @if(o_textures[i])
            @{s = o_clamp[i][0]}
            @{t = o_clamp[i][1]}

            vec2 texSize@{i} = TEX_SIZE(@{i});

            @if(!s && !t)
                vec2 vTexCoordAdj@{i} = vTexCoord@{i};
            @else
                @if(s && t)
                    vec2 vTexCoordAdj@{i} = clamp(vTexCoord@{i}, 0.5 / texSize@{i}, vec2(vTexClampS@{i}, vTexClampT@{i}));
                @elseif(s)
                    vec2 vTexCoordAdj@{i} = vec2(clamp(vTexCoord@{i}.s, 0.5 / texSize@{i}.s, vTexClampS@{i}), vTexCoord@{i}.t);
                @else
                    vec2 vTexCoordAdj@{i} = vec2(vTexCoord@{i}.s, clamp(vTexCoord@{i}.t, 0.5 / texSize@{i}.t, vTexClampT@{i}));
                @end
            @end

            vec4 texVal@{i} = hookTexture2D(@{i}, uTex@{i}, vTexCoordAdj@{i}, texSize@{i});

            @if(o_masks[i])
                @if(opengles) 
                    vec2 maskSize@{i} = vec2(textureSize(uTexMask@{i}, 0));
                @else 
                    vec2 maskSize@{i} = textureSize(uTexMask@{i}, 0);
                @end

                vec4 maskVal@{i} = hookTexture2D(@{i}, uTexMask@{i}, vTexCoordAdj@{i}, maskSize@{i});

                @if(o_blend[i])
                    vec4 blendVal@{i} = hookTexture2D(@{i}, uTexBlend@{i}, vTexCoordAdj@{i}, texSize@{i});
                @else
                    vec4 blendVal@{i} = vec4(0, 0, 0, 0);
                @end

                texVal@{i} = mix(texVal@{i}, blendVal@{i}, maskVal@{i}.a);
            @end
        @end
    @end

    @if(o_shadow_solid)
        // Reconstruct a solid silhouette from the 96x96 coverage mask with a one-screen-pixel transition.
        // Derivative scaling keeps the contour clean under magnification, minification and perspective without
        // paying the CPU/upload cost of a larger mask.
        float shadowEdgeWidth = max(fwidth(texVal0.a) * 0.75, 1.0 / 255.0);
        texVal0.a = smoothstep(0.5 - shadowEdgeWidth, 0.5 + shadowEdgeWidth, texVal0.a);
        if (texVal0.a <= 1.0 / 255.0) discard;
    @end

    @if(o_alpha) 
        vec4 texel;
    @else 
        vec3 texel;
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
            texel = vec4(@{
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

    texel = WRAP(texel, -0.51, 1.51);
    texel = clamp(texel, 0.0, 1.0);
    // TODO discard if alpha is 0?

    @if(o_water)
        float waterTime = water_time_seconds;
        vec2 waterBaseUv = vWorldPos.xz * max(water_normal_scale, 0.00001);
        vec2 waterUv1 = waterBaseUv + water_uv_speed1 * waterTime;
        vec2 waterUv2 = vec2(-waterBaseUv.y, waterBaseUv.x) + water_uv_speed2 * waterTime;

        vec2 waterSlope1;
        @if(o_textures[0])
            vec2 waterTexel1 = 1.0 / max(vec2(texture_width[0], texture_height[0]), vec2(1.0));
            float waterH1 = dot(@{texture}(uTex0, waterUv1).rgb, vec3(0.299, 0.587, 0.114));
            waterSlope1 = vec2(
                dot(@{texture}(uTex0, waterUv1 + vec2(waterTexel1.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114)) - waterH1,
                dot(@{texture}(uTex0, waterUv1 + vec2(0.0, waterTexel1.y)).rgb, vec3(0.299, 0.587, 0.114)) - waterH1);
        @else
            waterSlope1 = vec2(cos(waterUv1.x * 6.283), sin(waterUv1.y * 6.283)) * 0.08;
        @end

        vec2 waterSlope2;
        @if(o_textures[1])
            vec2 waterTexel2 = 1.0 / max(vec2(texture_width[1], texture_height[1]), vec2(1.0));
            float waterH2 = dot(@{texture}(uTex1, waterUv2).rgb, vec3(0.299, 0.587, 0.114));
            waterSlope2 = vec2(
                dot(@{texture}(uTex1, waterUv2 + vec2(waterTexel2.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114)) - waterH2,
                dot(@{texture}(uTex1, waterUv2 + vec2(0.0, waterTexel2.y)).rgb, vec3(0.299, 0.587, 0.114)) - waterH2);
        @elseif(o_textures[0])
            vec2 waterTexel2 = 1.0 / max(vec2(texture_width[0], texture_height[0]), vec2(1.0));
            float waterH2 = dot(@{texture}(uTex0, waterUv2).rgb, vec3(0.299, 0.587, 0.114));
            waterSlope2 = vec2(
                dot(@{texture}(uTex0, waterUv2 + vec2(waterTexel2.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114)) - waterH2,
                dot(@{texture}(uTex0, waterUv2 + vec2(0.0, waterTexel2.y)).rgb, vec3(0.299, 0.587, 0.114)) - waterH2);
        @else
            waterSlope2 = vec2(sin(waterUv2.x * 5.17), cos(waterUv2.y * 7.11)) * 0.08;
        @end

        vec3 waterDx = dFdx(vWorldPos);
        vec3 waterDy = dFdy(vWorldPos);
        vec3 waterGeometricN = normalize(cross(waterDx, waterDy));
        vec3 waterV = normalize(water_camera_pos - vWorldPos);
        if (dot(waterGeometricN, waterV) < 0.0) waterGeometricN = -waterGeometricN;
        vec3 waterTangent = normalize(waterDx);
        vec3 waterBitangent = normalize(cross(waterGeometricN, waterTangent));
        vec2 waterSlope = (waterSlope1 + waterSlope2) * max(water_normal_strength, 0.0) * 6.0;
        vec3 waterN = normalize(waterGeometricN - waterTangent * waterSlope.x - waterBitangent * waterSlope.y);

        vec2 waterScreenUv = gl_FragCoord.xy / max(water_viewport_size, vec2(1.0));
        float waterDepthGap = max(water_fade_distance, 1.0);
        if (water_depth_available > 0.5) {
            float waterSceneZ = @{texture}(water_scene_depth, clamp(waterScreenUv, 0.0, 1.0)).r;
            waterDepthGap = max(waterLinearDepth(waterSceneZ) - waterLinearDepth(gl_FragCoord.z), 0.0);
        }
        float waterDeepFactor = smoothstep(0.0, max(water_fade_distance, 1.0), waterDepthGap);
        vec4 waterDepthTint = mix(water_shallow_color, water_deep_color, waterDeepFactor);
        vec4 waterBaseColor = texel;
        waterBaseColor.rgb *= waterDepthTint.rgb;
        waterBaseColor.a = waterDepthTint.a;

        vec2 waterCausticUv = vWorldPos.xz * max(water_caustic_scale, 0.00001);
        vec2 waterCausticWarp = vec2(
            sin(waterCausticUv.y * 1.71 + waterTime * 0.73) +
                sin(waterCausticUv.x * 0.67 - waterTime * 0.41),
            cos(waterCausticUv.x * 1.43 - waterTime * 0.61) +
                cos(waterCausticUv.y * 0.79 + waterTime * 0.53)) * 0.16;
        waterCausticUv += waterCausticWarp + water_uv_speed1 * waterTime * 8.0;
        float waterCaustic = waterCausticWeb(waterCausticUv, water_caustic_thickness);

        float waterFresnel = pow(1.0 - clamp(dot(waterN, waterV), 0.0, 1.0),
                                 max(water_fresnel_power, 0.01));
        vec2 waterMirrorUv = vec2(waterScreenUv.x, 1.0 - waterScreenUv.y);
        vec2 waterReflectUv = clamp(waterMirrorUv + vec2(waterN.x, -waterN.z) * water_reflection_distortion,
                                    vec2(0.002), vec2(0.998));
        vec3 waterSceneReflection = @{texture}(water_scene_color, waterReflectUv).rgb;
        float waterReflectionAmount = clamp(mix(0.20, 1.0, waterFresnel) * water_reflection_intensity,
                                            0.0, 1.0);
        waterBaseColor.rgb = mix(waterBaseColor.rgb, waterSceneReflection,
                                 waterReflectionAmount);

        float waterCausticFacing = mix(1.0, 0.65, waterFresnel);
        float waterCausticAmount = clamp(waterCaustic * water_caustic_strength *
                                         water_caustic_color.a * waterCausticFacing, 0.0, 1.0);
        waterBaseColor = mix(waterBaseColor, water_caustic_color, waterCausticAmount);

        vec3 waterL = normalize(water_light_dir);
        vec3 waterH = normalize(waterL + waterV);
        float waterSpecular = step(clamp(water_specular_threshold, 0.0, 1.0),
                                   max(dot(waterN, waterH), 0.0));
        waterSpecular *= step(0.0, dot(waterN, waterL)) * max(water_specular_intensity, 0.0);
        waterBaseColor.rgb += water_light_color * waterSpecular;

        float waterFoam = 0.0;
        if (water_depth_available > 0.5) {
            float waterContact = 1.0 - smoothstep(0.0, max(water_foam_thickness, 0.001), waterDepthGap);
            float waterIrregular = 1.0 - waterCaustic;
            waterFoam = clamp(waterContact * mix(0.45, 1.0, waterIrregular), 0.0, 1.0);
        }
        waterBaseColor = mix(waterBaseColor, water_foam_color, waterFoam * water_foam_color.a);
        texel = clamp(waterBaseColor, 0.0, 1.0);
    @end

    // SOH [Enhancement] Toon lighting: re-light the (white-shaded) albedo with the single
    // dominant light through a soft half-Lambert ramp. Wind Waker-style two-tone.
    @if(o_toon)
        vec3 albedoColor = texel.rgb;
        vec3 toonN = normalize(vNormal);
        vec3 toonL = normalize(toon_light_dir);
        float toonNL = dot(toonN, toonL) * 0.5 + 0.5;
        float toonRamp = smoothstep(toon_ramp_center - toon_ramp_softness,
                                    toon_ramp_center + toon_ramp_softness, toonNL);
        vec3 toonLit = toon_ambient + toon_light_color * toon_highlight_intensity;
        vec3 toonShadow = mix(toonLit, toon_ambient, toon_shadow_intensity);
        if (toon_debug > 0.5) {
            // Diagnostic view: flat white on the lit side of the ramp, flat black in shadow, albedo
            // discarded â makes it obvious which draws are receiving toon lighting.
            texel.rgb = vec3(toonRamp);
        } else {
            texel.rgb = clamp(texel.rgb * mix(toonShadow, toonLit, toonRamp), 0.0, 1.0);
            // Stylized, light-directed inner silhouette band for opaque lit actor materials only. The
            // opt_toon variant is armed around actors and requires G_LIGHTING; !o_alpha rejects translucent,
            // particle-like draws. Sky, water, UI and ordinary scenery never compile this path.
            @if(!o_alpha)
                if (toon_rim_enabled > 0.5) {
                    vec3 viewDelta = toon_camera_pos - vWorldPos;
                    vec3 V = viewDelta * inversesqrt(max(dot(viewDelta, viewDelta), 0.000001));
                    float facing = clamp(dot(toonN, V), 0.0, 1.0);
                    // Convert angular facing into an approximate screen-pixel distance from the silhouette.
                    // A nearly side-facing triangle with little variation therefore no longer lights up as a
                    // whole face, which is the characteristic failure of a plain Fresnel threshold.
                    float facingGradient = max(fwidth(facing), 0.0001);
                    float silhouetteDistancePixels = facing / facingGradient;
                    float widthControl = clamp(toon_rim_width, 0.0, 1.0);
                    float rimWidthPixels = mix(0.25, 1.0, widthControl * widthControl);
                    float smoothControl = clamp(toon_rim_softness / 0.15, 0.0, 1.0);
                    float featherPixels = mix(0.30, 0.65, smoothControl);
                    float rimBand = 1.0 - smoothstep(rimWidthPixels - featherPixels,
                                                     rimWidthPixels + featherPixels,
                                                     silhouetteDistancePixels);

                    float lightSide = smoothstep(-0.35, 0.10, dot(toonN, toonL));
                    float backLighting = smoothstep(0.0, 0.65, dot(-V, toonL));
                    float directionalMask = lightSide * mix(0.25, 1.0, backLighting);
                    // Keep a directional component even at the lowest user setting: the effect must never
                    // collapse into a uniform Fresnel halo around the whole actor.
                    float directionInfluence = clamp(toon_rim_direction_influence, 0.25, 1.0);
                    directionalMask = mix(1.0, directionalMask, directionInfluence);

                    // This shader variant itself is the material mask: opaque + lit + actor-toon eligible.
                    // Emissive/unlit and transparent materials never reach this branch.
                    float materialRimMask = 1.0;
                    float rimMask = clamp(rimBand * directionalMask * materialRimMask *
                                          max(toon_rim_intensity, 0.0), 0.0, 1.0);
                    // A mostly pale key-light tint matches the reference's bright outline instead of merely
                    // lifting the material albedo across a wide side-facing region.
                    vec3 rimColor = mix(vec3(1.0), clamp(toon_light_color, 0.0, 1.0), 0.35);
                    texel.rgb = mix(texel.rgb, rimColor, rimMask);
                }
            @end
        }
    @end

    @if(o_fog)
        @if(o_alpha)
            texel = vec4(mix(texel.rgb, vFog.rgb, vFog.a), texel.a);
        @else
            texel = mix(texel, vFog.rgb, vFog.a);
        @end
    @end

    @if(o_texture_edge && o_alpha)
        if (texel.a > 0.19) texel.a = 1.0; else discard;
    @end

    @if(o_alpha && o_noise)
        texel.a *= floor(clamp(random(vec3(floor(gl_FragCoord.xy * noise_scale), float(frame_count))) + texel.a, 0.0, 1.0));
    @end

    @if(o_grayscale)
        float intensity = (texel.r + texel.g + texel.b) / 3.0;
        vec3 new_texel = vGrayscaleColor.rgb * intensity;
        texel.rgb = mix(texel.rgb, new_texel, vGrayscaleColor.a);
    @end

    @if(o_alpha)
        @if(o_alpha_threshold)
            if (texel.a < 8.0 / 256.0) discard;
        @end
        @if(o_invisible)
            texel.a = 0.0;
        @end
        @{vOutColor} = texel;
    @else
        @{vOutColor} = vec4(texel, 1.0);
    @end

    @if(srgb_mode)
        @{vOutColor} = fromLinear(@{vOutColor});
    @end
}
