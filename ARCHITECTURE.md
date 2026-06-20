# FaceMask Expressive System Architecture

## 1. Project goal

Build an offline, low-latency facial expression proxy system for a wearable mask/helmet.

The system runs on Raspberry Pi 4B with Python 3.13 and converts camera-observed facial motion into a stable expression state and a pixel-face output that can later drive LED hardware.

This is not a face-recognition project. It is a control-signal system:

- extract lightweight facial signals
- smooth them into stable internal states
- render those states into a visible pixel-face representation

The architecture should support:

- close-range camera placement inside a dark mask
- wide-angle lens distortion
- infrared fill lighting / night-vision conditions
- partial-face visibility in MVP
- future upgrade to full-face and multimodal input
- future output to LED matrix hardware

## 2. Design principles

- Prefer geometry over AI.
- Prefer control over prediction.
- Prefer stability over accuracy.
- Prefer modular interfaces over camera-specific code.
- Prefer continuous control signals plus stable discrete states.
- Prefer one shared runtime pipeline with pluggable sensors, trackers, and renderers.

## 3. Scope split

### MVP scope

MVP should reliably estimate expression-like output from the upper-face region only, or from whatever facial region is visible inside the mask.

MVP outputs:

- current `expression_state`
- current continuous feature values
- rendered pixel face in terminal / debug window / image dump
- detailed logs for tuning

### Post-MVP expansion

Later versions may support:

- full-face tracking
- microphone-derived features
- head-motion features
- LED hardware output
- richer expression vocabulary
- user calibration profiles

## 4. Recommended architecture

```text
face_mask/
├── main.py
├── config/
│   ├── default.yaml
│   ├── mock_upper_face.yaml
│   ├── mock_full_face.yaml
│   └── pi_nightvision.yaml
├── core/
│   ├── pipeline.py
│   ├── types.py
│   ├── state_machine.py
│   ├── smoothing.py
│   └── logging_setup.py
├── sensors/
│   ├── base.py
│   ├── mock_sensor.py
│   ├── camera_sensor.py
│   └── replay_sensor.py
├── tracking/
│   ├── base.py
│   ├── upper_face_tracker.py
│   ├── full_face_tracker.py
│   └── roi_manager.py
├── features/
│   ├── eye_feature.py
│   ├── brow_feature.py
│   ├── visibility_feature.py
│   ├── lighting_feature.py
│   └── fusion.py
├── renderers/
│   ├── base.py
│   ├── preset_renderer.py
│   ├── procedural_renderer.py
│   └── ascii_renderer.py
├── output/
│   ├── debug_output.py
│   └── led_driver.py
└── utils/
    ├── timing.py
    └── image_ops.py
```

## 5. Core pipeline

The runtime should be a single shared pipeline:

```text
Sensor -> Tracker -> Feature Extractors -> Smoothing -> State Machine -> Renderer -> Output
```

Recommended per-frame flow:

1. Capture a frame from the selected sensor.
2. Normalize the frame for the current capture mode.
3. Estimate the visible facial region.
4. Extract continuous features.
5. Smooth features over time.
6. Update stable state with hysteresis / debounce.
7. Render a pixel-face frame.
8. Emit debug/log/output events.

This keeps future upgrades localized. Camera changes should not require renderer rewrites. Renderer changes should not require tracker rewrites.

## 6. Separate the two key modes

Do not use a single mode switch for everything.

Use two independent axes:

### Tracking mode

Controls what facial region is being interpreted.

Supported values:

- `upper_face`
- `full_face`
- `adaptive_visible_region`
- later: `multimodal`

`adaptive_visible_region` is important for the mask use case. The camera may only see brows, one eye, both eyes, or an uneven crop. The runtime should tolerate partial visibility and degrade gracefully.

### Render mode

Controls how the output face is produced.

Supported values:

- `preset`
- `procedural`

This separation keeps future extensions clean:

- `upper_face + preset`
- `upper_face + procedural`
- `full_face + procedural`
- `multimodal + preset`

## 7. Data model

Use a small, explicit typed data model across the pipeline.

### Frame packet

```python
FramePacket:
    frame_id: int
    timestamp_monotonic: float
    image: np.ndarray
    gray: np.ndarray | None
    metadata: dict
```

### Tracking result

```python
TrackingResult:
    tracking_mode: str
    visible_regions: dict[str, ROI]
    visibility_score: float
    face_present: bool
    lighting_score: float
    notes: list[str]
```

### Feature bundle

```python
FeatureBundle:
    eye_open: float | None
    brow_raise: float | None
    mouth_open: float | None
    asymmetry: float | None
    motion_level: float | None
    visibility_score: float
    lighting_score: float
    confidence: float
```

### Expression output

```python
ExpressionOutput:
    expression_state: str
    intensity: float
    confidence: float
    render_variant: str | None
    debug_tags: list[str]
```

This structure allows missing values cleanly. For MVP, `mouth_open` can remain `None` without contaminating the rest of the pipeline.

## 8. Sensor layer design

### `MockSensor`

Used on Mac and during architecture tuning.

Should support multiple mock profiles:

- `cycle_states`: loop through states on a timer
- `wave_features`: sinusoidal `eye_open` / `brow_raise`
- `scripted_sequence`: YAML/JSON timeline of feature changes
- `replay_frames`: play back saved images or recorded clips
- `noise_injection`: add jitter, dropouts, low-light noise

Recommended mock config knobs:

- FPS
- random seed
- feature noise amplitude
- visibility dropout rate
- lighting fluctuation amplitude
- scripted hold durations

### `CameraSensor`

Should expose only camera-specific responsibilities:

- open camera
- read frame
- apply capture settings
- return `FramePacket`

Configuration should include:

- camera index or device path
- target width/height
- target FPS
- grayscale or color mode
- horizontal flip / vertical flip
- exposure / gain if supported
- IR mode flag
- optional lens calibration profile

### `ReplaySensor`

Useful for deterministic debugging on desktop and Pi.

Reads saved frames or short clips and replays them through the full runtime. This is the best way to debug state-machine behavior without needing to wear the hardware every run.

## 9. Dark mask / night vision / wide-angle considerations

This is an important constraint and should shape the implementation from the start.

### 9.1 Low light inside the mask

Likely issues:

- low contrast in eyebrow region
- high sensor noise
- unstable auto exposure
- non-uniform illumination from IR emitters
- partial overexposure near LEDs

Recommended handling:

- prefer grayscale-first processing
- normalize local contrast before extracting features
- log mean and variance of ROI brightness
- detect exposure instability and down-weight noisy features
- keep per-feature confidence values

### 9.2 Infrared illumination

IR changes image characteristics. Some skin and eyebrow details may appear very different from visible-light expectations.

Recommended handling:

- avoid color-based logic for MVP
- design feature extractors around intensity, edges, geometry, temporal change
- add a config switch like `illumination_mode = visible | infrared`
- allow different thresholds per illumination mode

### 9.3 160-degree wide-angle lens

Wide-angle distortion can warp eyebrows and eye shape, especially near image edges.

Recommended handling:

- keep ROI selection centered when possible
- support optional lens calibration / undistortion profile
- never hardcode absolute geometry assumptions too early
- derive features from relative motion and local region structure, not precise global face proportions

### 9.4 Very close camera placement

Close range changes apparent motion a lot:

- small facial movement can cause large pixel displacement
- partial occlusion is common
- the whole face may not fit in frame

Recommended handling:

- make `adaptive_visible_region` a first-class tracking mode
- separate `face_present` from `full_face_available`
- allow one-eye / partial-brow operation instead of hard failure

## 10. Feature extraction strategy

For MVP, start with lightweight and explainable features.

### Eye openness

Priority order:

1. simple local geometry or contour proxy
2. edge-density / dark-light separation proxy
3. EAR-like approximation if landmarks are feasible without heavy dependencies

Output:

- normalized `eye_open` in `[0, 1]`
- optional left/right values for future asymmetry support

### Brow raise

Start with a proxy rather than semantic recognition.

Candidate methods:

- eyebrow ROI brightness distribution change
- edge profile shift in upper-eye region
- eyebrow-to-eye relative gap estimate where possible
- temporal difference against rolling baseline

Output:

- normalized `brow_raise` in `[-1, 1]`

### Visibility and lighting

These are not optional. They should be treated as real features.

Compute:

- `visibility_score`
- `lighting_score`
- `motion_level`
- `confidence`

When these degrade, the state machine should become conservative instead of flickering.

## 11. State machine design

The state machine should consume continuous signals and output stable state.

Suggested MVP states:

- `neutral`
- `sleepy`
- `surprised`
- `cheerful`
- optional later: `tense`

Use three mechanisms together:

### 11.1 Temporal smoothing

Use EMA or short rolling windows for feature smoothing.

Examples:

- fast smoothing for `eye_open`
- slower smoothing for `brow_raise`
- extra smoothing when lighting is unstable

### 11.2 Hysteresis

Use different enter/exit thresholds for each state.

Example:

- enter `sleepy` when `eye_open < 0.28`
- exit `sleepy` only when `eye_open > 0.38`

This prevents rapid oscillation.

### 11.3 Minimum hold time

Require a new state to remain likely for a minimum duration before switching.

Example:

- `sleepy` can trigger quickly
- `surprised` may require stronger evidence but shorter hold
- `neutral` may require stabilization after high-motion intervals

The state machine should also support an internal `unknown` or `low_confidence_hold` mode even if that is not exposed to the visible renderer.

## 12. Rendering strategy

Both rendering modes are worth keeping.

### 12.1 Preset renderer

Each state has multiple pixel variants.

Example:

- `cheerful`: 4 variants
- `surprised`: 3 variants
- `sleepy`: 3 variants

Variant selection can be based on:

- current intensity
- recent motion
- random seed with cooldown
- asymmetry or brow level

This gives visual variety without losing control.

### 12.2 Procedural renderer

The renderer should generate each frame from:

- `expression_state`
- `eye_open`
- `brow_raise`
- `intensity`
- later: `mouth_open`, `audio_energy`

This mode is the long-term target because it preserves continuous expressiveness and maps well to LED output.

### 12.3 Shared rendering contract

Both renderers should implement the same interface:

```python
render(expression_output, feature_bundle, style_config) -> PixelFrame
```

That allows easy switching during runtime.

## 13. Output modes

Recommended early output targets:

- terminal state-change logging
- ASCII face debug renderer
- OpenCV debug preview when GUI is available
- image-frame dump for offline inspection

Future output:

- LED matrix driver
- serial / SPI / I2C hardware transport

The renderer should produce an abstract `PixelFrame`. Hardware output should consume that frame separately.

## 14. Mock and debug plan

This should be built in from day one.

### Essential mock profiles

#### `mock_upper_face.yaml`

Simulates normal upper-face usage:

- eye blink cycles
- occasional brow raise
- moderate noise
- stable lighting

#### `mock_lowlight_ir.yaml`

Simulates mask interior with IR light:

- low contrast
- brightness hotspots
- moderate sensor noise
- visibility dips

#### `mock_occlusion.yaml`

Simulates partial face visibility:

- one brow missing
- one eye partially clipped
- shifting ROI confidence

#### `mock_state_stress.yaml`

Exercises the state machine:

- rapid threshold crossings
- jitter near state boundaries
- exposure swings
- short motion bursts

### Replay fixtures

Store short controlled recordings for regression checks:

- neutral baseline
- repeated blinks
- brow raise sequence
- low-light noisy sequence
- partial-face sequence

These fixtures should be replayable on both Mac and Pi.

## 15. Logging and observability

Logging should be much richer than typical demo scripts.

Use structured logging with clear event categories.

### Log categories

- `runtime.startup`
- `sensor.frame`
- `tracking.result`
- `features.values`
- `features.confidence`
- `state.transition`
- `renderer.variant`
- `perf.frame_time`
- `perf.fps`
- `warning.visibility_drop`
- `warning.lighting_instability`

### Per-frame debug payload

At debug level, log:

- frame id
- timestamp
- tracking mode
- visible regions
- eye_open
- brow_raise
- confidence
- visibility score
- lighting score
- current state
- candidate state
- selected render mode
- render variant
- frame processing time

### Log outputs

Recommended outputs:

- console human-readable logs
- rotating file logs
- optional JSONL log for later analysis

### Why this matters

This project will require threshold tuning under unusual camera conditions. Without strong logs, debugging will be guesswork.

## 16. Performance guidance for Raspberry Pi 4B

Target:

- 15 to 30 FPS
- low jitter
- CPU usage under control

Recommended practices:

- process grayscale unless color is proven necessary
- crop ROIs early
- avoid expensive full-frame transforms every tick
- support configurable processing resolution
- decouple capture FPS from heavy debug rendering when needed
- measure every stage timing
- make debug image dumps optional

Useful perf metrics:

- capture time
- tracking time
- feature extraction time
- state machine time
- rendering time
- total frame time
- effective FPS

## 17. Configuration strategy

Use config files rather than scattered constants.

Recommended top-level config keys:

```yaml
runtime:
  tracking_mode: upper_face
  render_mode: procedural
  log_level: DEBUG
  fps_target: 20

sensor:
  type: camera
  camera_index: 0
  width: 640
  height: 480
  infrared_mode: true
  lens_profile: wide160

tracking:
  adaptive_visible_region: true
  min_visibility_score: 0.35

features:
  eye_smoothing_alpha: 0.45
  brow_smoothing_alpha: 0.25

state_machine:
  min_hold_ms: 180
  sleepy_enter: 0.28
  sleepy_exit: 0.38

renderer:
  resolution: 16
  preset_variant_count: 4
```

The code should accept config-path override from CLI.

## 18. Testing strategy

### Unit-level

Test:

- smoothing behavior
- hysteresis thresholds
- state transitions
- renderer output dimensions
- renderer variant selection

### Integration-level

Test pipeline behavior with:

- `MockSensor`
- `ReplaySensor`
- fixed seed randomization

### Scenario regression

Keep a few deterministic replay runs and verify:

- expected transitions happen
- no flicker under boundary noise
- low-confidence periods do not thrash states

## 19. Recommended implementation order

### Phase 1

- config loader
- typed pipeline data model
- `MockSensor`
- basic state machine
- preset renderer
- ASCII / terminal output
- structured logging

### Phase 2

- procedural renderer
- `CameraSensor`
- grayscale preprocessing
- adaptive visible-region tracking
- replay sensor

### Phase 3

- brow feature extraction improvements
- low-light / IR tuning
- state-machine tuning with replay fixtures
- perf tuning on Pi

### Phase 4

- full-face mode
- optional audio feature channel
- LED driver abstraction

## 20. Recommended best-practice decisions

- Keep camera IO, feature extraction, state logic, and rendering in separate modules.
- Represent missing data explicitly with `None` plus confidence scores.
- Make low-confidence behavior conservative rather than expressive.
- Make all thresholds configurable.
- Build deterministic replay tooling early.
- Avoid static emoji assets as a core dependency.
- Keep hardware output behind a stable `PixelFrame` abstraction.
- Treat IR mode and visible-light mode as separate operating conditions.

## 21. Questions that still matter before implementation

The current architecture is ready for review, but these decisions will affect implementation details:

1. Final camera interface on Pi: USB UVC, CSI, or another board.
2. Planned LED resolution and wiring constraints.
3. Whether debug preview on Pi needs GUI output or only headless logs/files.
4. Whether you want state names optimized for external appearance (`cheerful`) or emotional framing (`happy`).

## 22. Recommended next step

Implement the MVP around this exact split:

- `tracking_mode = upper_face`
- `render_mode = preset | procedural`
- `sensor = mock first, camera second`
- `visibility/confidence` as first-class signals
- replayable mocks and structured logs from the start

That gives you a Pi-friendly architecture that works for the dark close-range mask setup now and still expands cleanly to full-face, microphone, and LED output later.

## 23. Normal camera tryout flow

For early bring-up with a standard visible-light camera, start with the dedicated config:

- `face_mask/config/pi_visible_camera.yaml`

Run:

- `python3 -m face_mask.main --config face_mask/config/pi_visible_camera.yaml`

What to verify first:

- camera stream opens reliably
- both eyes and brow area are centered enough for the static ROI layout
- `eye_open`, `brow_raise`, `visibility_score`, and `lighting_score` move in plausible directions
- state changes are stable and do not flicker excessively

Important interpretation note:

- current tracking is heuristic and calibration-driven, not identity-aware face recognition
- it does not require enrolling a user face or defining a personal template in advance
- it still depends on runtime calibration of ROI placement, lighting, and thresholds for the physical camera setup
