@prism(type='fragment', name='Fast3D Fragment Shader', version='1.0.0', description='Ported shader to prism', author='Emill & Prism Team')

@{GLSL_VERSION}

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
@if(o_toon) @{attr} vec3 vWorldPos;

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
        // Actor-shadow coverage is deliberately thresholded after bilinear filtering. The texture itself is
        // binary, but bilinear sampling otherwise reintroduces a noisy low-alpha fringe at the silhouette.
        if (texVal0.a < 0.5) discard;
        texVal0.a = 1.0;
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
                    float edge = 1.0 - clamp(dot(toonN, V), 0.0, 1.0);
                    // Treat Width as a perceptual control: most of its travel is reserved for a very thin
                    // silhouette contour, while the top end can still produce a broader stylized rim.
                    float widthControl = clamp(toon_rim_width, 0.0, 1.0);
                    float rimWidth = mix(0.02, 0.22, widthControl * widthControl);
                    float threshold = 1.0 - rimWidth;
                    // Smoothness now scales screen-space antialiasing instead of being hidden behind a fixed
                    // 3.5% floor. This makes every part of the UI range visibly affect the transition without
                    // changing the contour's geometric width.
                    float smoothControl = clamp(toon_rim_softness / 0.15, 0.0, 1.0);
                    float feather = max(fwidth(edge), 0.0005) * mix(0.55, 4.0, smoothControl);
                    float rimBand = smoothstep(threshold - feather, threshold + feather, edge);

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
