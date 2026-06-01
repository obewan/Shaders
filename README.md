# Obewan Shaders

Post-processing shaders for ReShade, aimed at a realistic image — primarily for *The Elder Scrolls V: Skyrim Special Edition*.

## ReshadeTrueLight

An all-in-one realistic lighting & grading pipeline in a single technique (**v1.0.0**):

**Lighting & geometry**
- Ambient occlusion — SSAO (fast) and GTAO (quality) modes, per-pixel rotated sampling, distance fade + distant-radius controls
- Contact shadows
- Screen-space reflections — Fresnel-masked, adaptive sphere-trace march, binary refinement, metallic + glossy/roughness
- Depth of field — blurs the fully processed image (composite-based), separable bokeh, focus-aware sharpen

**Atmosphere & light**
- Bloom — progressive dual-filter pyramid (CoD/Jimenez) with soft-knee threshold
- God rays — volumetric light shafts from an estimated on-screen light position
- Lens flare — screen-space ghosts + halo with chromatic aberration
- Distance / aerial fog — with sun forward-scattering glow and night darkening

**Tone & colour**
- Auto-exposure with smooth, frame-rate-independent eye adaptation
- Tonemap operators — AgX (default), ACES, Hable (Uncharted 2), Reinhard
- Grading — contrast, saturation, temperature/tint white balance
- Local contrast / clarity
- Purkinje night-vision (the eye's rod/scotopic shift in darkness)

**Finishing**
- Contrast-adaptive sharpening (CAS)
- Luminance-aware film grain, blue-noise option, dithering

Controls are organised into collapsible categories in the ReShade overlay. Typical cost is ~1–2 fps.

### Requirements

- A working depth buffer (verify with the stock `DisplayDepth` shader).
- `bluenoise.png` in `reshade-shaders/textures/` — only if **Use Blue Noise** is enabled.

### Anti-aliasing

ReshadeTrueLight intentionally does **not** include AA — a dedicated AA shader does it better. Enable ReShade's built-in **SMAA** (no extra download) and order it **before** ReshadeTrueLight in the technique list, so edges are cleaned before the lighting/sharpen work. **CShade DLAA** is a good alternative if you prefer it. Avoid Skyrim's native TAA when using the depth-based effects — its per-frame jitter makes the depth buffer wobble and adds noise to AO/SSR.

### Installation

Copy the contents of `reshade-shaders/` into your ReShade `reshade-shaders/` folder, then enable **ReshadeTrueLight** in the ReShade overlay.

## Compatibility

- **Community Shaders** — fully compatible, and the recommended pairing. Community Shaders handles engine-level lighting and GI; ReshadeTrueLight adds the screen-space post (reflections, god rays, fog, grading and the rest) on top.
- **ENBSeries** — ENB and ReShade *can* run together, but it isn't recommended with this shader: ENB already provides its own ambient occlusion, bloom, depth of field and tonemapping, so the effects would double up. Use one or the other.

Note: ENB and Community Shaders are mutually exclusive (you run one or the other), but ReshadeTrueLight, being a ReShade shader, works alongside either.

## Notes & limitations

ReShade only has access to the final colour image and the depth buffer — not the engine's G-buffer. So effects that need real surface materials or light data (true GI, world-space height fog, sun-accurate shadows) are intentionally left to engine-level tools like **Community Shaders** or **ENBSeries**; ReshadeTrueLight focuses on the screen-space post that those don't fully cover.

## License

See repository.

## Acknowledgements

Developed with the help of AI assistants (Anthropic Claude, and earlier scaffolding from Microsoft Copilot, OpenAI ChatGPT and Google Gemini) for code review, debugging and implementation. The design decisions, testing and final tuning are the author's own.
