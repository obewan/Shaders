//===========================================================================
// SKYRIM REALISTIC PIPELINE — ReshadeTrueLight Header
// Author: Obewan (https://github.com/obewan)
// Version: 1.2.0
// Requirements
//   - bluenoise.png in reshade-shaders/textures if Use Blue Noise = true
//   - A working depth buffer (verify with the stock DisplayDepth shader)
//
// NOTE: temporal reprojection has been removed. Stock ReShade cannot
// feed per-frame view/projection matrices into a uniform, so the previous
// reprojection path could never work without a custom C++ addon. All effects
// are now single-frame and rely on spatial bilateral filtering for stability.
//
// v1.2.0 added the "Distant Shading" UI category (fake directional shading for
// geometry past the engine's shadow distance) and a matching Debug View entry.
//
// v1.1.0 added two UI categories - "Indirect Light" (screen-space colour
// bleeding, gathered by the AO pass) and "Light Source" (the shared brightest-
// on-screen-light estimate that god rays, lens flare, fog glow and contact
// shadows all read). The AO render targets changed from R8 to RGBA16F to carry
// the bounce alongside the occlusion. See the .fx header for the full list.
//===========================================================================

#include "ReShadeUI.fxh"
#include "ReShade.fxh"

// Compile-time bounds for bounded loops
static const int AO_MAX_SAMPLES = 12;
static const int DOF_MAX_TAPS   = 9;
static const int SSR_MAX_STEPS  = 32;

// Precomputed unit directions for AO sampling
static const float2 AO_Dirs[AO_MAX_SAMPLES] =
{
    float2( 1.000000,  0.000000),
    float2( 0.866025,  0.500000),
    float2( 0.500000,  0.866025),
    float2( 0.000000,  1.000000),
    float2(-0.500000,  0.866025),
    float2(-0.866025,  0.500000),
    float2(-1.000000,  0.000000),
    float2(-0.866025, -0.500000),
    float2(-0.500000, -0.866025),
    float2( 0.000000, -1.000000),
    float2( 0.500000, -0.866025),
    float2( 0.866025, -0.500000)
};

// Precomputed 9-tap Gaussian weights (center first)
static const float DOF_WEIGHTS[DOF_MAX_TAPS] =
{
    0.027027f, 0.054054f, 0.121621f, 0.242422f, 0.333333f,
    0.242422f, 0.121621f, 0.054054f, 0.027027f
};

// ============================
// FRAME INFO (engine-driven, hidden — no UI bar)
// ============================
uniform int FrameIndex < source = "framecount"; >;

// ============================
// DEPTH / CAMERA
// ReShade::GetLinearizedDepth returns normalized [0,1] (1 = far plane / sky).
// CameraFar scales that into the world units used for position reconstruction.
// CameraFovY replaces the (unavailable) projection matrix for unprojection.
// ============================
uniform bool  HasDepth  < ui_category = "Depth & Camera"; ui_type = "checkbox"; ui_label = "Has Depth"; ui_tooltip = "Uncheck if the game exposes no usable depth buffer."; > = true;
uniform float CameraFar  < ui_category = "Depth & Camera"; ui_type = "slider"; ui_min = 100.0; ui_max = 10000.0; ui_step = 10.0; ui_label = "Camera Far (depth scale)"; ui_tooltip = "World-unit scale applied to normalized depth."; > = 1000.0;
uniform float CameraFovY < ui_category = "Depth & Camera"; ui_type = "slider"; ui_min = 30.0;  ui_max = 120.0;   ui_step = 1.0;  ui_label = "Vertical FOV (deg)"; ui_tooltip = "Match the game's vertical field of view."; > = 60.0;

// ============================
// TEXTURES / SAMPLERS
// ============================
texture2D BackBufferTex : COLOR;
sampler BackBuffer { Texture = BackBufferTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// Bloom pyramid (progressive down/up-sample, CoD/Jimenez style). RGBA16F, LINEAR.
#define BLOOM_SAMP(s, t) sampler s { Texture = t; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D BloomD0 { Width = BUFFER_WIDTH/2;  Height = BUFFER_HEIGHT/2;  Format = RGBA16F; }; BLOOM_SAMP(BloomD0s, BloomD0)
texture2D BloomD1 { Width = BUFFER_WIDTH/4;  Height = BUFFER_HEIGHT/4;  Format = RGBA16F; }; BLOOM_SAMP(BloomD1s, BloomD1)
texture2D BloomD2 { Width = BUFFER_WIDTH/8;  Height = BUFFER_HEIGHT/8;  Format = RGBA16F; }; BLOOM_SAMP(BloomD2s, BloomD2)
texture2D BloomD3 { Width = BUFFER_WIDTH/16; Height = BUFFER_HEIGHT/16; Format = RGBA16F; }; BLOOM_SAMP(BloomD3s, BloomD3)
texture2D BloomD4 { Width = BUFFER_WIDTH/32; Height = BUFFER_HEIGHT/32; Format = RGBA16F; }; BLOOM_SAMP(BloomD4s, BloomD4)
texture2D BloomD5 { Width = BUFFER_WIDTH/64; Height = BUFFER_HEIGHT/64; Format = RGBA16F; }; BLOOM_SAMP(BloomD5s, BloomD5)
texture2D BloomU0 { Width = BUFFER_WIDTH/2;  Height = BUFFER_HEIGHT/2;  Format = RGBA16F; }; BLOOM_SAMP(BloomU0s, BloomU0)
texture2D BloomU1 { Width = BUFFER_WIDTH/4;  Height = BUFFER_HEIGHT/4;  Format = RGBA16F; }; BLOOM_SAMP(BloomU1s, BloomU1)
texture2D BloomU2 { Width = BUFFER_WIDTH/8;  Height = BUFFER_HEIGHT/8;  Format = RGBA16F; }; BLOOM_SAMP(BloomU2s, BloomU2)
texture2D BloomU3 { Width = BUFFER_WIDTH/16; Height = BUFFER_HEIGHT/16; Format = RGBA16F; }; BLOOM_SAMP(BloomU3s, BloomU3)
texture2D BloomU4 { Width = BUFFER_WIDTH/32; Height = BUFFER_HEIGHT/32; Format = RGBA16F; }; BLOOM_SAMP(BloomU4s, BloomU4)

// Luminance downsample (used for instant auto-exposure)
texture2D LumCurrTex { Width = 64; Height = 64; Format = R16F; };
sampler LumCurrSampler { Texture = LumCurrTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// Adapted luminance (1x1, smoothed over time via a tiny ping-pong — the only
// cross-frame state in the shader; needs no reprojection matrices).
texture2D AdaptTex { Width = 1; Height = 1; Format = R16F; };
sampler AdaptSampler { Texture = AdaptTex; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D AdaptPrevTex { Width = 1; Height = 1; Format = R16F; };
sampler AdaptPrevSampler { Texture = AdaptPrevTex; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// Light position: brightest-area centroid (rg = screen UV, b = confidence), for
// god rays / lens flares. 1x1 with a ping-pong for temporal smoothing.
texture2D LightPosTex { Width = 1; Height = 1; Format = RGBA16F; }; BLOOM_SAMP(LightPosSampler, LightPosTex)
texture2D LightPosPrevTex { Width = 1; Height = 1; Format = RGBA16F; }; BLOOM_SAMP(LightPosPrevSampler, LightPosPrevTex)

// God rays: occluder-masked half-res bright source + radial scatter result.
texture2D GodrayBrightTex { Width = BUFFER_WIDTH/2; Height = BUFFER_HEIGHT/2; Format = RGBA16F; }; BLOOM_SAMP(GodrayBrightSampler, GodrayBrightTex)
texture2D GodrayTex       { Width = BUFFER_WIDTH/2; Height = BUFFER_HEIGHT/2; Format = RGBA16F; }; BLOOM_SAMP(GodraySampler, GodrayTex)

// Lens flare (ghosts + halo) — reuses the bloom prefilter as its bright source.
texture2D LensFlareTex    { Width = BUFFER_WIDTH/2; Height = BUFFER_HEIGHT/2; Format = RGBA16F; }; BLOOM_SAMP(LensFlareSampler, LensFlareTex)

// AO (computed at half-res, upsampled in the blur).
// rgb = indirect bounce radiance, a = occlusion. RGBA16F because the bounce is
// linear-light and would band badly in 8-bit at the low end (dungeon interiors).
texture2D AORawTex { Width = BUFFER_WIDTH/2; Height = BUFFER_HEIGHT/2; Format = RGBA16F; };
sampler AORawSampler { Texture = AORawTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D AOBlurTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };
sampler AOBlurSampler { Texture = AOBlurTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// Contact shadows (full-res)
texture2D ContactRawTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R8; };
sampler ContactRawSampler { Texture = ContactRawTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D ContactBlurTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R8; };
sampler ContactBlurSampler { Texture = ContactBlurTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// SSR (computed at half-res, upsampled in the blur)
texture2D SSRRawTex { Width = BUFFER_WIDTH/2; Height = BUFFER_HEIGHT/2; Format = RGBA8; };
sampler SSRRawSampler { Texture = SSRRawTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D SSRBlurTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler SSRBlurSampler { Texture = SSRBlurTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// DoF (separable: horizontal gather -> vertical gather). rgb = blurred color, a = normalized CoC.
texture2D DoFRawTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler DoFRawSampler { Texture = DoFRawTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D DoFBlurTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler DoFBlurSampler { Texture = DoFBlurTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// Fully processed scene (AO/contact/exposure/SSR/bloom/tonemap) BEFORE DoF, so
// DoF blurs the finished image instead of the raw backbuffer.
texture2D CompositeTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler CompositeSampler { Texture = CompositeTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// ============================
// BLUE-NOISE
// Place BlueNoise.png in reshade-shaders/Textures. Default off to avoid load failures.
// ============================
texture2D BlueNoiseTex < source = "bluenoise.png"; pooled = true; > { Width = 64; Height = 64; Format = RGBA8; };
sampler BlueNoiseSampler { Texture = BlueNoiseTex; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = REPEAT; AddressV = REPEAT; };

// ============================
// EXPOSURE
// ============================
uniform float frametime < source = "frametime"; >; // milliseconds since last frame
uniform bool  EnableAutoExposure < ui_category = "Exposure"; ui_type = "checkbox"; ui_label = "Auto Exposure"; > = true;
uniform float ExposureKey  < ui_category = "Exposure"; ui_type = "slider"; ui_min = 0.05; ui_max = 0.50; ui_step = 0.01; ui_label = "Exposure Key (middle grey)"; > = 0.18;
uniform float AdaptSpeed   < ui_category = "Exposure"; ui_type = "slider"; ui_min = 0.1;  ui_max = 8.0;  ui_step = 0.1;  ui_label = "Adaptation Speed"; ui_tooltip = "Higher = faster eye adaptation. Lower = smoother."; > = 2.0;
uniform float ExposureMin  < ui_category = "Exposure"; ui_type = "slider"; ui_min = 0.05; ui_max = 2.0;  ui_step = 0.01; ui_label = "Min Exposure (gain floor)"; > = 0.25;
uniform float ExposureMax  < ui_category = "Exposure"; ui_type = "slider"; ui_min = 0.5;  ui_max = 8.0;  ui_step = 0.05; ui_label = "Max Exposure (gain ceiling)"; ui_tooltip = "Lower this to stop dark/night scenes from over-brightening."; > = 2.0;
uniform float ManualExposure < ui_category = "Exposure"; ui_type = "slider"; ui_min = -4.0; ui_max = 4.0; ui_step = 0.1; ui_label = "Manual Exposure (EV, when auto off)"; > = 0.0;
uniform bool  MeteringLogAverage   < ui_category = "Exposure"; ui_type = "checkbox"; ui_label = "Log-average Metering"; ui_tooltip = "Meter on the geometric mean instead of the arithmetic mean, so a bright sun or torch in frame stops dragging the whole scene dark - the usual cause of 'everything dims when I look at the sky'.\n\nIf the image now sits brighter than you had it tuned, lower Exposure Key a little."; > = true;
uniform float MeteringCenterWeight < ui_category = "Exposure"; ui_type = "slider"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01; ui_label = "Centre-weighted Metering"; ui_tooltip = "0 = meter the whole frame evenly. Higher = weight the centre of the screen, so exposure follows what you are looking at rather than a bright sky in the corner."; > = 0.35;

// ============================
// TONEMAP & GRADING
// ============================
uniform int   TonemapOperator < ui_category = "Tonemap & Grading"; ui_type = "combo"; ui_items = "ACES\0AgX\0Hable (Uncharted 2)\0Reinhard\0"; ui_label = "Tonemap Operator"; ui_tooltip = "Maps HDR-range light to a displayable image; each has a different look.\n\nACES: punchy, high contrast, cinematic, but skews bright saturated colours toward orange/white (hue shift).\n\nAgX: most colour-accurate at extremes - fire, sunsets and magic keep their hue instead of blowing to white. Flatter, neutral, modern default.\n\nHable (Uncharted 2): filmic toe/shoulder curve, cinematic contrast; often the most photographic. White Point is meaningful.\n\nReinhard: softest and flattest, a neutral reference. White Point only nudges the brightest highlights."; > = 1;
uniform float TonemapWhite    < ui_category = "Tonemap & Grading"; ui_type = "slider"; ui_min = 1.0; ui_max = 16.0; ui_step = 0.1; ui_label = "White Point (Hable/Reinhard)"; ui_tooltip = "Brightness that maps to pure white. On Hable it shifts the whole curve (very visible); on Reinhard it only affects the brightest highlights. Ignored by ACES/AgX."; > = 4.0;
uniform float Contrast        < ui_category = "Tonemap & Grading"; ui_type = "slider"; ui_min = 0.5; ui_max = 2.0;  ui_step = 0.01; ui_label = "Contrast"; ui_tooltip = "Tonal contrast around mid-grey (luminance-based, so it won't grey out or clip colours). Lower (~0.85) to flatten AgX/ACES punch for a softer, more filmic look."; > = 1.0;
uniform float Saturation      < ui_category = "Tonemap & Grading"; ui_type = "slider"; ui_min = 0.0; ui_max = 2.0;  ui_step = 0.01; ui_label = "Saturation"; ui_tooltip = "Colour saturation. Nudge up (~1.1) to compensate for the flatness of AgX or a lowered Contrast."; > = 1.0;
uniform float WhiteTemp       < ui_category = "Tonemap & Grading"; ui_type = "slider"; ui_min = -100.0; ui_max = 100.0; ui_step = 1.0; ui_label = "Temperature"; ui_tooltip = "White balance: warm/orange (+) to cool/blue (-). Luminance-preserving."; > = 0.0;
uniform float WhiteTint       < ui_category = "Tonemap & Grading"; ui_type = "slider"; ui_min = -100.0; ui_max = 100.0; ui_step = 1.0; ui_label = "Tint"; ui_tooltip = "White balance: magenta (+) to green (-). Luminance-preserving."; > = 0.0;

// ============================
// PURKINJE (night vision) — the eye's rod/scotopic shift in darkness
// ============================
uniform bool  EnablePurkinje    < ui_category = "Night Vision (Purkinje)"; ui_type = "checkbox"; ui_label = "Enable Purkinje (Night Vision)"; ui_tooltip = "Human vision desaturates and shifts blue in darkness (rod vision). Engages at night / in dungeons, off in daylight. It's eye physiology, not a camera effect."; > = true;
uniform float PurkinjeStrength  < ui_category = "Night Vision (Purkinje)"; ui_type = "slider"; ui_min = 0.0;  ui_max = 1.0;  ui_step = 0.01; ui_label = "Purkinje Strength"; > = 0.5;
uniform float PurkinjeThreshold < ui_category = "Night Vision (Purkinje)"; ui_type = "slider"; ui_min = 0.01; ui_max = 0.30; ui_step = 0.01; ui_label = "Purkinje Threshold"; ui_tooltip = "Scene darkness (adapted luminance) below which night vision kicks in. Higher = engages in brighter scenes."; > = 0.10;

// ============================
// AO
// ============================
uniform bool  EnableAO       < ui_category = "Ambient Occlusion"; ui_type = "checkbox"; ui_label = "Enable AO"; > = true;
uniform int   AOMode         < ui_category = "Ambient Occlusion"; ui_type = "combo"; ui_items = "SSAO (fast, low-end)\0GTAO (quality)\0"; ui_label = "AO Mode"; ui_tooltip = "SSAO: cheap disk sampling. GTAO: horizon-marched, more accurate, a bit heavier."; > = 0;
uniform float AORadius       < ui_category = "Ambient Occlusion"; ui_type = "slider"; ui_min = 0.05; ui_max = 20.0; ui_step = 0.05; ui_label = "AO Radius (world)"; > = 3.0;
uniform int   AOSamples      < ui_category = "Ambient Occlusion"; ui_type = "slider"; ui_min = 4;    ui_max = 12;   ui_step = 1;    ui_label = "AO Samples"; > = 12;
uniform float AOStrength     < ui_category = "Ambient Occlusion"; ui_type = "slider"; ui_min = 0.0;  ui_max = 4.0;  ui_step = 0.01; ui_label = "AO Strength"; > = 1.6;
uniform float AOPower        < ui_category = "Ambient Occlusion"; ui_type = "slider"; ui_min = 0.5;  ui_max = 4.0;  ui_step = 0.05; ui_label = "AO Contrast (power)"; ui_tooltip = "Higher = deeper, more contrasty occlusion."; > = 2.0;
uniform float AOBias         < ui_category = "Ambient Occlusion"; ui_type = "slider"; ui_min = 0.0;  ui_max = 0.5;  ui_step = 0.005; ui_label = "AO Bias"; ui_tooltip = "Raise to reduce self-occlusion / flat-surface noise."; > = 0.03;
uniform float AOFadeStart    < ui_category = "Ambient Occlusion"; ui_type = "slider"; ui_min = 0.0;  ui_max = 1.0;  ui_step = 0.01; ui_label = "AO Fade Start (depth)"; ui_tooltip = "Normalized depth where AO begins to fade out."; > = 0.6;
uniform float AOFadeEnd      < ui_category = "Ambient Occlusion"; ui_type = "slider"; ui_min = 0.0;  ui_max = 1.0;  ui_step = 0.01; ui_label = "AO Fade End (depth)"; ui_tooltip = "Normalized depth where AO is fully gone. Lower this if distant terrain (mountains) still shows AO noise."; > = 0.9;
uniform float AODistantRadius < ui_category = "Ambient Occlusion"; ui_type = "slider"; ui_min = 0.0; ui_max = 0.1; ui_step = 0.002; ui_label = "AO Distant Radius"; ui_tooltip = "Grows the AO radius with distance so far objects keep a stable, less-noisy AO footprint (XeGTAO-style). 0 = off. An alternative to fading AO out."; > = 0.0;

// ============================
// INDIRECT LIGHT (screen-space colour bleeding / one bounce)
// ============================
uniform bool  EnableGI     < ui_category = "Indirect Light"; ui_type = "checkbox"; ui_label = "Enable Indirect Light"; ui_tooltip = "One-bounce screen-space colour bleeding, gathered by the AO pass. Nearby lit surfaces throw their colour back into the cavities AO just darkened: sunlit rock warms the ground beside it, a torch spills orange onto the wall it faces.\n\nNot true GI - ReShade has no G-buffer or light data, so the on-screen lit colour stands in for the bounce source. Engine-level GI is Community Shaders' job; this fills in the screen-space part."; > = true;
uniform float GIStrength   < ui_category = "Indirect Light"; ui_type = "slider"; ui_min = 0.0; ui_max = 2.0; ui_step = 0.01; ui_label = "Indirect Strength"; ui_tooltip = "How much bounced light comes back. Raise until crevices and interiors stop reading as flat black holes; back off if the image starts to glow."; > = 0.6;
uniform float GISaturation < ui_category = "Indirect Light"; ui_type = "slider"; ui_min = 0.0; ui_max = 2.0; ui_step = 0.01; ui_label = "Indirect Saturation"; ui_tooltip = "Colour intensity of the bounce. 0 = neutral grey fill light that only lifts the shadows. 1 = the neighbour's actual colour. Above 1 exaggerates the colour bleed."; > = 1.0;

// ============================
// CONTACT SHADOWS
// ============================
uniform bool  EnableContact   < ui_category = "Contact Shadows"; ui_type = "checkbox"; ui_label = "Enable Contact Shadows"; > = true;
uniform float ContactMaxDist  < ui_category = "Contact Shadows"; ui_type = "slider"; ui_min = 0.05; ui_max = 5.0; ui_step = 0.05; ui_label = "Contact Max Distance (world)"; > = 1.0;
uniform float ContactStrength < ui_category = "Contact Shadows"; ui_type = "slider"; ui_min = 0.0;  ui_max = 1.0; ui_step = 0.01; ui_label = "Contact Strength"; > = 0.65;

// ============================
// DISTANT SHADING
// Stands in for the shadowing the engine stops doing past its shadow distance.
// AO cannot cover this: its radius shrinks with distance, so it only ever
// darkens contact-scale creases, never a whole mountain face.
// ============================
uniform bool  EnableDistantShade    < ui_category = "Distant Shading"; ui_type = "checkbox"; ui_label = "Enable Distant Shading"; ui_tooltip = "Fake directional shading for far geometry. Skyrim's shadow map ends well before the horizon, so distant LOD gets sun + ambient with no occlusion and reads as a flat, bright cut-out - the usual cause of 'it's too bright at distance even with AO on'. This shades it by the angle to the tracked light instead, so hillsides facing away go dark again.\n\nUse Debug View > Distant Shading to see the term on its own."; > = true;
uniform float DistantShadeStart     < ui_category = "Distant Shading"; ui_type = "slider"; ui_min = 0.0; ui_max = 500.0; ui_step = 1.0; ui_label = "Start Distance (world)"; ui_tooltip = "Where the effect begins to fade in. Keep this past the engine's own shadow distance or you'll double-shade the near field. Same world units as Fog Start and DoF Focal Distance (normalized depth x Camera Far)."; > = 60.0;
uniform float DistantShadeFull      < ui_category = "Distant Shading"; ui_type = "slider"; ui_min = 1.0; ui_max = 1000.0; ui_step = 5.0; ui_label = "Full Distance (world)"; ui_tooltip = "Where it reaches full strength. A wide gap between this and Start Distance hides the transition."; > = 300.0;
uniform float DistantShadeStrength  < ui_category = "Distant Shading"; ui_type = "slider"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01; ui_label = "Shading Strength"; ui_tooltip = "How dark a surface facing fully away from the light goes. Raise until distant mountains regain their form; back off if they start reading as flat black."; > = 0.35;
uniform float DistantShadeWrap      < ui_category = "Distant Shading"; ui_type = "slider"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01; ui_label = "Terminator Softness"; ui_tooltip = "Wrapped diffuse. 0 = hard light/shade line. Higher wraps light around the form, which is what haze at distance actually does. Lower it for crisp, high-sun terrain."; > = 0.5;
uniform float DistantShadeScale     < ui_category = "Distant Shading"; ui_type = "slider"; ui_min = 1.0; ui_max = 16.0; ui_step = 0.5; ui_label = "Detail Scale (px)"; ui_tooltip = "Pixel baseline for the normal estimate. Small = per-facet detail but noisy at range, since one pixel of far-field parallax is below the depth buffer's precision. Larger reads the broad landform and is much steadier. Raise this first if distant terrain shimmers."; > = 4.0;
uniform float DistantHazeFade       < ui_category = "Distant Shading"; ui_type = "slider"; ui_min = 0.0; ui_max = 1000.0; ui_step = 5.0; ui_label = "Haze Fadeout (world)"; ui_tooltip = "Distance at which the shading starts fading back out again, reaching zero at Camera Far. Past a point most of what reaches you from a mountain is in-scattered atmosphere, not light off rock - and haze is additive, so darkening it doesn't shade the mountain, it dims the air in front of it and the horizon just looks grimy. Set this to roughly where haze starts dominating your view (Debug View > Depth, then multiply by Camera Far). 0 = never fade out."; > = 600.0;
uniform float DistantHighlightProtect < ui_category = "Distant Shading"; ui_type = "slider"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01; ui_label = "Highlight Protection"; ui_tooltip = "Spares surfaces the engine already drew bright. This pass fakes a shadow test, and a distant peak rendered that hot is one the sun is genuinely hitting - shading it doesn't restore form, it deletes the glare off sunlit snow. Raise if snowcaps go dull; lower if bright distant rock stays too flat. 0 = shade everything by normal alone (the old behaviour)."; > = 0.7;
uniform float DistantHighlightKnee    < ui_category = "Distant Shading"; ui_type = "slider"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01; ui_label = "Highlight Knee"; ui_tooltip = "Screen brightness at which protection starts fading in, reaching full at white. Measured display-referred, so the number matches what you see rather than the much smaller linear value. Lower it to protect more of the midtones, raise it so only true highlights - snow, water glint - are spared. Capped at 0.9 internally: at 1.0 the ramp would be degenerate and would silently switch protection off entirely."; > = 0.55;
uniform float DistantAmbientDim     < ui_category = "Distant Shading"; ui_type = "slider"; ui_min = 0.0; ui_max = 0.5; ui_step = 0.01; ui_label = "Distant Ambient Dim"; ui_tooltip = "Flat brightness reduction at distance, on top of the directional term. Unlike the shading it needs no light direction, so it still works overcast or at night. A little goes a long way - this is the knob for 'the whole horizon is too bright'."; > = 0.10;

// ============================
// SSR
// ============================
uniform bool  EnableSSR       < ui_category = "Screen-Space Reflections"; ui_type = "checkbox"; ui_label = "Enable SSR"; > = true;
uniform int   SSRSteps        < ui_category = "Screen-Space Reflections"; ui_type = "slider"; ui_min = 8;    ui_max = 32;    ui_step = 1;   ui_label = "SSR Steps"; > = 20;
uniform float SSRMaxDistance  < ui_category = "Screen-Space Reflections"; ui_type = "slider"; ui_min = 1.0;  ui_max = 400.0; ui_step = 1.0; ui_label = "SSR Max Distance (world)"; > = 60.0;
uniform float SSRStrength     < ui_category = "Screen-Space Reflections"; ui_type = "slider"; ui_min = 0.0;  ui_max = 1.0;   ui_step = 0.01; ui_label = "SSR Strength"; > = 0.5;
uniform float SSRBaseReflect  < ui_category = "Screen-Space Reflections"; ui_type = "slider"; ui_min = 0.0;  ui_max = 0.2;   ui_step = 0.005; ui_label = "SSR Base Reflectivity (F0)"; ui_tooltip = "Head-on reflectivity. Keep low (~0.02) so only grazing angles reflect — this stops the 'mirror on NPC' look."; > = 0.02;
uniform float SSRThickness    < ui_category = "Screen-Space Reflections"; ui_type = "slider"; ui_min = 0.5;  ui_max = 50.0;  ui_step = 0.5; ui_label = "SSR Thickness (world)"; ui_tooltip = "Reject hits where the scene lies far behind the ray (silhouette bleed). Raise if reflections vanish, lower if backgrounds smear into reflections."; > = 8.0;
uniform float SSRMetallic     < ui_category = "Screen-Space Reflections"; ui_type = "slider"; ui_min = 0.0;  ui_max = 1.0;   ui_step = 0.01; ui_label = "SSR Metallic"; ui_tooltip = "0 = dielectric (clear reflection). 1 = metal (reflection tinted by the surface colour, stronger Fresnel)."; > = 0.0;
uniform float SSRGlossiness   < ui_category = "Screen-Space Reflections"; ui_type = "slider"; ui_min = 0.0;  ui_max = 1.0;   ui_step = 0.01; ui_label = "SSR Glossiness"; ui_tooltip = "1 = sharp mirror reflection. Lower = rougher / blurrier reflection. (Heuristic; ReShade cannot read TruePBR roughness.)"; > = 0.7;

// ============================
// DoF (focal range in world units; CoC mapped to pixel radius)
// ============================
uniform bool  EnableDoF      < ui_category = "Depth of Field"; ui_type = "checkbox"; ui_label = "Enable DoF"; > = true;
uniform float FocalDistance  < ui_category = "Depth of Field"; ui_type = "slider"; ui_min = 0.0;  ui_max = 500.0; ui_step = 1.0; ui_label = "Focal Distance (world)"; > = 30.0;
uniform float FocalRange     < ui_category = "Depth of Field"; ui_type = "slider"; ui_min = 1.0;  ui_max = 300.0; ui_step = 1.0; ui_label = "In-focus Range (world)"; > = 40.0;
uniform float MaxCoCRadius   < ui_category = "Depth of Field"; ui_type = "slider"; ui_min = 0.5;  ui_max = 16.0;  ui_step = 0.25; ui_label = "Max CoC Radius (px)"; ui_tooltip = "Maximum blur radius in pixels. Small values (1-3) give subtle defocus; raise for stronger bokeh."; > = 3.0;

// ============================
// BLOOM
// ============================
uniform bool  EnableBloom    < ui_category = "Bloom"; ui_type = "checkbox"; ui_label = "Enable Bloom"; > = true;
uniform float BloomThreshold < ui_category = "Bloom"; ui_type = "slider"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01; ui_label = "Bloom Threshold (linear)"; ui_tooltip = "Brightness where bloom starts."; > = 0.7;
uniform float BloomSoftKnee  < ui_category = "Bloom"; ui_type = "slider"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01; ui_label = "Bloom Soft Knee"; ui_tooltip = "Soft transition around the threshold (0 = hard cutoff, 1 = very soft). Reduces flicker/popping on bright edges."; > = 0.5;
uniform float BloomStrength  < ui_category = "Bloom"; ui_type = "slider"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.005; ui_label = "Bloom Strength"; > = 0.08;
uniform float BloomRadius    < ui_category = "Bloom"; ui_type = "slider"; ui_min = 0.5; ui_max = 2.0; ui_step = 0.05; ui_label = "Bloom Spread"; ui_tooltip = "Width of the glow (scales the upsample tent)."; > = 1.0;

// ============================
// LIGHT SOURCE TRACKING
// The brightest on-screen area, estimated once per frame. Drives god rays, lens
// flare, fog forward-scattering and the contact-shadow light direction.
// ============================
uniform float LightPeakBias       < ui_category = "Light Source"; ui_type = "slider"; ui_min = 0.0; ui_max = 0.99; ui_step = 0.01; ui_label = "Peak Lock"; ui_tooltip = "How tightly the estimate locks onto the single brightest spot. A plain brightness centroid lands halfway between the sun and a bright snowfield - on neither. Higher = only cells near peak brightness count, so the sun wins. Lower it if the light position twitches between two similar sources."; > = 0.75;
uniform float LightTrackAmount    < ui_category = "Light Source"; ui_type = "slider"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01; ui_label = "Contact Shadow Sun Tracking"; ui_tooltip = "Point contact shadows away from the tracked light instead of a fixed direction, so they swing with the sun through the day. 0 = the old fixed key-light direction. Falls back to fixed on its own when no confident light is on screen."; > = 1.0;
uniform float LightConfidenceGate < ui_category = "Light Source"; ui_type = "slider"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01; ui_label = "Confidence Gate"; ui_tooltip = "Fade god rays, lens flare and fog glow out when nothing genuinely bright is on screen, so you don't get sun shafts in a windowless dungeon. 0 = never gate (old behaviour), 1 = fully gated."; > = 0.5;

// ============================
// GOD RAYS (volumetric light scattering from the brightest on-screen light)
// ============================
uniform bool  EnableGodrays   < ui_category = "God Rays"; ui_type = "checkbox"; ui_label = "Enable God Rays"; > = true;
uniform float GodrayIntensity < ui_category = "God Rays"; ui_type = "slider"; ui_min = 0.0; ui_max = 2.0;  ui_step = 0.01;  ui_label = "God Ray Intensity"; > = 0.5;
uniform float GodrayThreshold < ui_category = "God Rays"; ui_type = "slider"; ui_min = 0.0; ui_max = 1.0;  ui_step = 0.01;  ui_label = "God Ray Threshold"; ui_tooltip = "Brightness a pixel needs to emit shafts. Lower = fuller shafts from more of the sky.\n\nAlso the threshold Light Source Tracking uses to find the sun, so it feeds lens flare, fog glow and contact-shadow direction too. Measured after exposure."; > = 0.2;
uniform float GodrayDensity   < ui_category = "God Rays"; ui_type = "slider"; ui_min = 0.1; ui_max = 1.0;  ui_step = 0.01;  ui_label = "God Ray Length"; ui_tooltip = "How far the shafts reach toward the light."; > = 0.6;
uniform float GodrayDecay     < ui_category = "God Rays"; ui_type = "slider"; ui_min = 0.8; ui_max = 0.99; ui_step = 0.005; ui_label = "God Ray Decay"; ui_tooltip = "Falloff along each shaft. Higher = longer."; > = 0.95;
uniform float GodraySkyBias    < ui_category = "God Rays"; ui_type = "slider"; ui_min = 0.0; ui_max = 64.0; ui_step = 1.0; ui_label = "God Ray Sky Bias"; ui_tooltip = "Higher = shafts come from the distant sky/sun, not nearby bright geometry (snowy mountains). 0 = any bright pixel. If god rays vanish when you raise this, your depth buffer doesn't mark the sky as far -> leave it at 0."; > = 0.0;

// ============================
// LENS FLARE (screen-space ghosts + halo from the brightest areas)
// ============================
uniform bool  EnableLensFlare    < ui_category = "Lens Flare"; ui_type = "checkbox"; ui_label = "Enable Lens Flare"; > = true;
uniform float LensFlareIntensity < ui_category = "Lens Flare"; ui_type = "slider"; ui_min = 0.0; ui_max = 2.0;  ui_step = 0.01; ui_label = "Lens Flare Intensity"; > = 0.4;
uniform int   LensFlareGhosts    < ui_category = "Lens Flare"; ui_type = "slider"; ui_min = 1;   ui_max = 8;    ui_step = 1;    ui_label = "Lens Flare Ghosts"; ui_tooltip = "Number of ghost reflections along the line through screen centre."; > = 4;
uniform float LensFlareDispersal < ui_category = "Lens Flare"; ui_type = "slider"; ui_min = 0.1; ui_max = 0.6;  ui_step = 0.01; ui_label = "Lens Flare Dispersal"; ui_tooltip = "Spacing of the ghosts."; > = 0.3;
uniform float LensFlareHalo      < ui_category = "Lens Flare"; ui_type = "slider"; ui_min = 0.0; ui_max = 0.8;  ui_step = 0.01; ui_label = "Lens Flare Halo Width"; > = 0.4;
uniform float LensFlareCA        < ui_category = "Lens Flare"; ui_type = "slider"; ui_min = 0.0; ui_max = 8.0;  ui_step = 0.1;  ui_label = "Lens Flare Chromatic Aberration"; > = 2.0;

// ============================
// FOG (distance / aerial perspective)
// ============================
uniform bool   EnableFog    < ui_category = "Fog"; ui_type = "checkbox"; ui_label = "Enable Fog"; > = true;
uniform float3 FogColor     < ui_category = "Fog"; ui_type = "color"; ui_label = "Fog Color"; > = float3(0.62, 0.69, 0.80);
uniform float  FogStart     < ui_category = "Fog"; ui_type = "slider"; ui_min = 0.0;  ui_max = 1500.0;  ui_step = 10.0;    ui_label = "Fog Start (world)"; ui_tooltip = "Distance before fog begins."; > = 250.0;
uniform float  FogDensity   < ui_category = "Fog"; ui_type = "slider"; ui_min = 0.0;  ui_max = 0.003;   ui_step = 0.0001; ui_label = "Fog Density"; ui_tooltip = "How quickly fog accumulates with distance. Subtle is best (a little goes a long way)."; > = 0.0005;
uniform float  FogMax       < ui_category = "Fog"; ui_type = "slider"; ui_min = 0.0;  ui_max = 1.0;     ui_step = 0.001;  ui_label = "Fog Max"; ui_tooltip = "Maximum fog opacity at the far distance."; > = 0.005;
uniform float  FogSunAmount < ui_category = "Fog"; ui_type = "slider"; ui_min = 0.0;  ui_max = 1.0;    ui_step = 0.01;   ui_label = "Fog Sun Glow"; ui_tooltip = "Forward scattering: fog brightens toward the sun (its on-screen position)."; > = 0.5;
uniform float3 FogSunColor  < ui_category = "Fog"; ui_type = "color"; ui_label = "Fog Sun Color"; > = float3(1.0, 0.85, 0.6);
uniform float  FogNightDim       < ui_category = "Fog"; ui_type = "slider"; ui_min = 0.0;  ui_max = 1.0;  ui_step = 0.01; ui_label = "Fog Night Darkness"; ui_tooltip = "How dark the fog gets in dark scenes (1 = no darkening; lower = darker night fog). Fixes bright fog washing out far mountains at night."; > = 0.25;
uniform float  FogNightThreshold < ui_category = "Fog"; ui_type = "slider"; ui_min = 0.01; ui_max = 0.30; ui_step = 0.01; ui_label = "Fog Night Threshold"; ui_tooltip = "Scene darkness (adapted luminance) below which the fog dims. Same gate as Purkinje."; > = 0.10;

// ============================
// LOCAL CONTRAST (clarity)
// ============================
uniform bool  EnableClarity < ui_category = "Local Contrast"; ui_type = "checkbox"; ui_label = "Enable Clarity"; > = true;
uniform float ClarityAmount < ui_category = "Local Contrast"; ui_type = "slider"; ui_min = 0.0; ui_max = 1.5; ui_step = 0.01; ui_label = "Clarity Amount"; ui_tooltip = "Local midtone contrast: pulls out texture detail (rock, fabric, foliage) for a crisp, near-HDR pop without global contrast's flatness."; > = 0.3;
uniform float ClarityRadius < ui_category = "Local Contrast"; ui_type = "slider"; ui_min = 8.0; ui_max = 96.0; ui_step = 1.0; ui_label = "Clarity Radius (px)"; ui_tooltip = "Scale of the local contrast. Larger = broader, softer pop."; > = 40.0;

// ============================
// CHROMATIC ABERRATION (lateral, edge-weighted lens dispersion)
// ============================
uniform bool  EnableCA   < ui_category = "Chromatic Aberration"; ui_type = "checkbox"; ui_label = "Enable Chromatic Aberration"; ui_tooltip = "Lateral RGB dispersion that grows toward the frame edges, like a real lens. The centre stays sharp. An optical effect, applied before grain/dither."; > = true;
uniform float CAStrength < ui_category = "Chromatic Aberration"; ui_type = "slider"; ui_min = 0.0; ui_max = 8.0; ui_step = 0.1; ui_label = "CA Strength (corner px)"; ui_tooltip = "Maximum red/blue split at the frame corners, in pixels (0 at the centre). Subtle is best (~1-2 px)."; > = 1.5;

// ============================
// SHARPEN & GRAIN
// ============================
uniform bool  EnableSharpen < ui_category = "Sharpen & Grain"; ui_type = "checkbox"; ui_label = "Enable Sharpen"; > = true;
uniform int   SharpenMode   < ui_category = "Sharpen & Grain"; ui_type = "combo"; ui_items = "CAS (edge-safe)\0Detail (local contrast)\0"; ui_label = "Sharpen Mode"; ui_tooltip = "CAS: AMD's contrast-adaptive sharpen. Backs off at edges to avoid haloes, but never boosts anything - faint texture stays faint, so what you mostly notice is crisper edges.\n\nDetail: divides the high-pass by local contrast, so weak texture is amplified up to the prominence of strong texture while outlines are pushed down. A detail enhancer rather than a sharpener - this is the one that makes rock and cloth read. Masks depth edges to kill haloes and composites as an overlay to preserve tonality. Wants a higher Sharpness than CAS (~0.5-0.8)."; > = 1;
uniform float Sharpness     < ui_category = "Sharpen & Grain"; ui_type = "slider"; ui_min = 0.0; ui_max = 1.0;  ui_step = 0.01;  ui_label = "Sharpness"; ui_tooltip = "Contrast-adaptive sharpening: boosts fine detail while sparing strong edges from haloes/crunch. Scales the detail linearly, so 0.5 really is half of 1.0."; > = 0.25;
uniform float SharpenRadius < ui_category = "Sharpen & Grain"; ui_type = "slider"; ui_min = 1.0; ui_max = 6.0;  ui_step = 0.1;   ui_label = "Sharpen Radius (px)"; ui_tooltip = "Which detail scale gets boosted. At 1 px this is a pixel-level edge sharpener, and with an AA pass running before this shader it mostly re-crisps the edges AA just resolved - so it reads as anti-aliasing being undone rather than as texture. Raise to 2-4 px to hit the scale surface texture actually lives at (rock grain, fabric weave, wood), which is what makes materials pop. Past ~4 px it turns chunky and illustrated; Clarity handles anything broader."; > = 2.0;
uniform float SharpenNearBoost < ui_category = "Sharpen & Grain"; ui_type = "slider"; ui_min = 1.0; ui_max = 6.0; ui_step = 0.1; ui_label = "Near Detail Boost"; ui_tooltip = "Widens the sharpen kernel on close surfaces, up to this multiple of Sharpen Radius. Sharpening works in screen space, but perspective does not: the road under your feet spreads one texel over many pixels, so a radius tuned for distant geometry finds no contrast there and appears to do nothing. This scales the kernel by 1/distance to track the projected texel size. 1.0 = off (uniform pixel radius). It only ever grows the radius, so distant detail is unaffected."; > = 2.0;
uniform float SharpenNearDist  < ui_category = "Sharpen & Grain"; ui_type = "slider"; ui_min = 1.0; ui_max = 200.0; ui_step = 1.0; ui_label = "Near Boost Distance"; ui_tooltip = "Distance at which the boost reaches 1x, i.e. where Sharpen Radius applies as written. Anything nearer gets a proportionally wider kernel, up to Near Detail Boost. Same world units as DoF Focal Distance (normalized depth x Camera Far)."; > = 40.0;
uniform float SharpenDefocusLimit < ui_category = "Sharpen & Grain"; ui_type = "slider"; ui_min = 0.0; ui_max = 16.0; ui_step = 0.25; ui_label = "Defocus Suppression (px)"; ui_tooltip = "Blur radius at which DoF fully suppresses sharpening (and Clarity). Sharpening a defocused region only adds ringing, but the suppression has to track the real blur: a preset that focuses far while keeping a small Max CoC Radius is barely defocused up close, and gating on raw CoC would switch the sharpener off there for no reason - making the DoF's focal distance secretly the sharpener's working range too. Set to 0 to decouple sharpening from DoF completely. Matching your Max CoC Radius reproduces the old all-or-nothing behaviour."; > = 3.0;
uniform bool  DetailLumaOnly < ui_category = "Sharpen & Grain"; ui_type = "checkbox"; ui_label = "Detail: Luma Only"; ui_tooltip = "Detail mode. Applies the detail as luminance, avoiding the coloured speckle a per-channel high-pass leaves behind - red/blue areas are the worst for it. Uncheck for slightly punchier colour detail at the risk of fringing."; > = true;
uniform float DetailDepthMask < ui_category = "Sharpen & Grain"; ui_type = "slider"; ui_min = 0.0; ui_max = 20000.0; ui_step = 100.0; ui_label = "Detail: Depth Edge Mask"; ui_tooltip = "Detail mode. Suppresses sharpening across depth discontinuities, where it only ever produces haloes. A flat receding floor has a linear depth ramp and is invisible to this, so grazing ground keeps its detail; only real silhouettes and sharp curvature are masked. Raise if objects get outlined, lower to 0 to disable."; > = 4000.0;
uniform float DetailFloor   < ui_category = "Sharpen & Grain"; ui_type = "slider"; ui_min = 0.0001; ui_max = 0.02; ui_step = 0.0001; ui_label = "Detail: Flat Area Suppression"; ui_tooltip = "Detail mode. Floor on the local-contrast divisor. Because the filter normalises by local contrast, a genuinely flat region (sky, fog, deep shadow) would have its dither and compression noise amplified to look like texture. Raise this if the sky gets grainy; lower it to pull detail out of very smooth surfaces."; > = 0.001;
uniform bool  EnableGrain   < ui_category = "Sharpen & Grain"; ui_type = "checkbox"; ui_label = "Enable Grain"; > = true;
uniform float GrainAmount   < ui_category = "Sharpen & Grain"; ui_type = "slider"; ui_min = 0.0; ui_max = 0.10; ui_step = 0.001; ui_label = "Grain Amount"; ui_tooltip = "Film grain strength. Luminance-aware, so it peaks in midtones and fades in shadows/highlights."; > = 0.02;
uniform float GrainSize     < ui_category = "Sharpen & Grain"; ui_type = "slider"; ui_min = 1.0; ui_max = 4.0;  ui_step = 0.1;   ui_label = "Grain Size"; ui_tooltip = "Coarseness of the grain (1 = per-pixel, higher = chunkier)."; > = 1.0;
uniform bool  UseBlueNoise  < ui_category = "Sharpen & Grain"; ui_type = "checkbox"; ui_label = "Use Blue Noise"; ui_tooltip = "Use the blue-noise texture for grain and dithering (and SSR jitter) instead of white-noise hash. Less perceptible, more film-like. Requires bluenoise.png in reshade-shaders/textures."; > = false;

// ============================
// DEBUG
// ============================
uniform int DebugMode < ui_category = "Debug"; ui_type = "combo"; ui_items = "Off\0Depth\0Normals\0ViewPos\0CoC\0AO\0Contact\0SSR\0Bloom\0Godrays\0Lens Flare\0Indirect Light\0Light Position\0Distant Shading\0"; ui_label = "Debug View"; > = 0;
