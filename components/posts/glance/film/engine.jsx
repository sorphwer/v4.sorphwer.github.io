/* BEGIN USAGE */
// engine.jsx — timeline engine. Exports: Stage, Sprite, TextSprite, ImageSprite,
//   RectSprite, VideoSprite, PlaybackBar, TimelineContext, SpriteContext,
//   useTime, useTimeline, useSprite, Easing, interpolate, animate, clamp.
//
//   <Stage width={1280} height={720} duration={10} background="#f5f5f5">
//     <Sprite start={0} end={3}>
//       <TextSprite text="Hello" x={100} y={300} size={72} color="#111" />
//     </Sprite>
//     <Sprite start={2} end={8}>
//       <ImageSprite src="hero.png" x={200} y={120} width={640} height={360} kenBurns />
//     </Sprite>
//   </Stage>
//
// Stage({width,height,duration,background,loop,autoplay,persistKey,startAt,paused}) —
//   fills its parent block and scales the canvas to fit; scrubber + play/pause;
//   once the stage has focus (click it): ←/→ seek, space, 0/Home reset.
//   Persists the playhead under `persistKey` unless `startAt` (seconds) is given.
//   `paused` pauses from outside and resumes on false only if it did the pausing.
//   The canvas is an <svg><foreignObject>.
// Sprite({start,end,keepMounted}) — mounts children only while playhead is in
//   [start,end]. Children read {localTime, progress, duration} via useSprite().
// useTime() → seconds; useTimeline() → {time,duration,playing,setTime,setPlaying}.
// TextSprite({text,x,y,size,color,font,weight,align,entryDur,exitDur}) — fades/scales in+out.
// ImageSprite({src,x,y,width,height,fit,radius,kenBurns,placeholder}) — same, with optional ken-burns.
// RectSprite({x,y,width,height,color,radius}) — solid box with entry/exit.
// VideoSprite({src,start,end,speed,style}) — looped <video> clip synced to the timeline.
// Easing.{linear,easeIn/Out/InOut Quad/Cubic/Quart/Quint/Expo/Back, …}
// interpolate([t0,t1,…],[v0,v1,…],ease?) → (t)=>v  — piecewise tween.
// animate({from,to,start,end,ease}) → (t)=>v  — single tween.
//
// Build scenes by composing Sprites inside Stage. Absolutely-position elements.
/* END USAGE */
// ─────────────────────────────────────────────────────────────────────────────
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'

const SANS = "'InterVariable', Inter, system-ui, sans-serif"
const MONO = 'JetBrains Mono, ui-monospace, SFMono-Regular, monospace'
const BAR_H = 44 // playback bar height

// ── Easing functions (hand-rolled, Popmotion-style) ─────────────────────────
// All easings take t ∈ [0,1] and return eased t ∈ [0,1] (may overshoot for back/elastic).
export const Easing = {
  linear: (t) => t,

  // Quad
  easeInQuad: (t) => t * t,
  easeOutQuad: (t) => t * (2 - t),
  easeInOutQuad: (t) => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t),

  // Cubic
  easeInCubic: (t) => t * t * t,
  easeOutCubic: (t) => --t * t * t + 1,
  easeInOutCubic: (t) => (t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1),

  // Quart
  easeInQuart: (t) => t * t * t * t,
  easeOutQuart: (t) => 1 - --t * t * t * t,
  easeInOutQuart: (t) => (t < 0.5 ? 8 * t * t * t * t : 1 - 8 * --t * t * t * t),

  // Expo
  easeInExpo: (t) => (t === 0 ? 0 : Math.pow(2, 10 * (t - 1))),
  easeOutExpo: (t) => (t === 1 ? 1 : 1 - Math.pow(2, -10 * t)),
  easeInOutExpo: (t) => {
    if (t === 0) return 0
    if (t === 1) return 1
    if (t < 0.5) return 0.5 * Math.pow(2, 20 * t - 10)
    return 1 - 0.5 * Math.pow(2, -20 * t + 10)
  },

  // Sine
  easeInSine: (t) => 1 - Math.cos((t * Math.PI) / 2),
  easeOutSine: (t) => Math.sin((t * Math.PI) / 2),
  easeInOutSine: (t) => -(Math.cos(Math.PI * t) - 1) / 2,

  // Back (overshoot)
  easeOutBack: (t) => {
    const c1 = 1.70158,
      c3 = c1 + 1
    return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2)
  },
  easeInBack: (t) => {
    const c1 = 1.70158,
      c3 = c1 + 1
    return c3 * t * t * t - c1 * t * t
  },
  easeInOutBack: (t) => {
    const c1 = 1.70158,
      c2 = c1 * 1.525
    return t < 0.5
      ? (Math.pow(2 * t, 2) * ((c2 + 1) * 2 * t - c2)) / 2
      : (Math.pow(2 * t - 2, 2) * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2
  },

  // Elastic
  easeOutElastic: (t) => {
    const c4 = (2 * Math.PI) / 3
    if (t === 0) return 0
    if (t === 1) return 1
    return Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1
  },
}

// ── Core interpolation helpers ──────────────────────────────────────────────

// Clamp a value to [min, max]
export const clamp = (v, min, max) => Math.max(min, Math.min(max, v))

// interpolate([0, 0.5, 1], [0, 100, 50], ease?) -> fn(t)
// Popmotion-style: linearly maps t across input keyframes to output values,
// with optional easing per segment (single fn or array of fns).
export function interpolate(input, output, ease = Easing.linear) {
  return (t) => {
    if (t <= input[0]) return output[0]
    if (t >= input[input.length - 1]) return output[output.length - 1]
    for (let i = 0; i < input.length - 1; i++) {
      if (t >= input[i] && t <= input[i + 1]) {
        const span = input[i + 1] - input[i]
        const local = span === 0 ? 0 : (t - input[i]) / span
        const easeFn = Array.isArray(ease) ? ease[i] || Easing.linear : ease
        const eased = easeFn(local)
        return output[i] + (output[i + 1] - output[i]) * eased
      }
    }
    return output[output.length - 1]
  }
}

// animate({from, to, start, end, ease})(t) — simpler single-segment tween.
// Returns `from` before `start`, `to` after `end`.
export function animate({ from = 0, to = 1, start = 0, end = 1, ease = Easing.easeInOutCubic }) {
  return (t) => {
    if (t <= start) return from
    if (t >= end) return to
    const local = (t - start) / (end - start)
    return from + (to - from) * ease(local)
  }
}

// ── Timeline context ────────────────────────────────────────────────────────

export const TimelineContext = createContext({ time: 0, duration: 10, playing: false })

export const useTime = () => useContext(TimelineContext).time
export const useTimeline = () => useContext(TimelineContext)

// ── Sprite ──────────────────────────────────────────────────────────────────
// Renders children only when the playhead is inside [start, end]. Provides
// a sub-context with `localTime` (seconds since start) and `progress` (0..1).
//
//   <Sprite start={2} end={5}>
//     {({ localTime, progress }) => <Thing x={progress * 100} />}
//   </Sprite>
//
// Or as a plain wrapper — children can call useSprite() themselves.

export const SpriteContext = createContext({ localTime: 0, progress: 0, duration: 0 })
export const useSprite = () => useContext(SpriteContext)

export function Sprite({ start = 0, end = Infinity, children, keepMounted = false }) {
  const { time } = useTimeline()
  const visible = time >= start && time <= end
  if (!visible && !keepMounted) return null

  const duration = end - start
  const localTime = Math.max(0, time - start)
  const progress = duration > 0 && isFinite(duration) ? clamp(localTime / duration, 0, 1) : 0

  const value = { localTime, progress, duration, visible }

  return (
    <SpriteContext.Provider value={value}>
      {typeof children === 'function' ? children(value) : children}
    </SpriteContext.Provider>
  )
}

// ── Sample sprite components ────────────────────────────────────────────────

// TextSprite: fades/slides text in on entry, holds, then fades out on exit.
// Props: text, x, y, size, color, font, entryDur, exitDur, align
export function TextSprite({
  text,
  x = 0,
  y = 0,
  size = 48,
  color = '#111',
  font = SANS,
  weight = 600,
  entryDur = 0.45,
  exitDur = 0.35,
  entryEase = Easing.easeOutBack,
  exitEase = Easing.easeInCubic,
  align = 'left',
  letterSpacing = '-0.01em',
}) {
  const { localTime, duration } = useSprite()
  const exitStart = Math.max(0, duration - exitDur)

  let opacity = 1
  let ty = 0

  if (localTime < entryDur) {
    const t = entryEase(clamp(localTime / entryDur, 0, 1))
    opacity = t
    ty = (1 - t) * 16
  } else if (localTime > exitStart) {
    const t = exitEase(clamp((localTime - exitStart) / exitDur, 0, 1))
    opacity = 1 - t
    ty = -t * 8
  }

  const translateX = align === 'center' ? '-50%' : align === 'right' ? '-100%' : '0'

  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: y,
        transform: `translate(${translateX}, ${ty}px)`,
        opacity,
        fontFamily: font,
        fontSize: size,
        fontWeight: weight,
        color,
        letterSpacing,
        whiteSpace: 'pre',
        lineHeight: 1.1,
        willChange: 'transform, opacity',
      }}
    >
      {text}
    </div>
  )
}

// ImageSprite: scales + fades in; optional Ken Burns drift during hold.
export function ImageSprite({
  src,
  x = 0,
  y = 0,
  width = 400,
  height = 300,
  entryDur = 0.6,
  exitDur = 0.4,
  kenBurns = false,
  kenBurnsScale = 1.08,
  radius = 12,
  fit = 'cover',
  placeholder = null, // {label: string} for striped placeholder
}) {
  const { localTime, duration } = useSprite()
  const exitStart = Math.max(0, duration - exitDur)

  let opacity = 1
  let scale = 1

  if (localTime < entryDur) {
    const t = Easing.easeOutCubic(clamp(localTime / entryDur, 0, 1))
    opacity = t
    scale = 0.96 + 0.04 * t
  } else if (localTime > exitStart) {
    const t = Easing.easeInCubic(clamp((localTime - exitStart) / exitDur, 0, 1))
    opacity = 1 - t
    scale = (kenBurns ? kenBurnsScale : 1) + 0.02 * t
  } else if (kenBurns) {
    const holdSpan = exitStart - entryDur
    const holdT = holdSpan > 0 ? (localTime - entryDur) / holdSpan : 0
    scale = 1 + (kenBurnsScale - 1) * holdT
  }

  const content = placeholder ? (
    <div
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'repeating-linear-gradient(135deg, #ebebeb 0 10px, #e0e0e0 10px 20px)',
        color: '#737373',
        fontFamily: 'JetBrains Mono, ui-monospace, monospace',
        fontSize: 13,
        letterSpacing: '0.04em',
        textTransform: 'uppercase',
      }}
    >
      {placeholder.label || 'image'}
    </div>
  ) : (
    <img
      src={src}
      alt=""
      style={{ width: '100%', height: '100%', objectFit: fit, display: 'block' }}
    />
  )

  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: y,
        width,
        height,
        opacity,
        transform: `scale(${scale})`,
        transformOrigin: 'center',
        borderRadius: radius,
        overflow: 'hidden',
        willChange: 'transform, opacity',
      }}
    >
      {content}
    </div>
  )
}

// RectSprite: simple rectangle that animates position/size/color via props.
// Useful demo primitive — takes a `render` fn for per-frame customization.
export function RectSprite({
  x = 0,
  y = 0,
  width = 100,
  height = 100,
  color = '#111',
  radius = 8,
  entryDur = 0.4,
  exitDur = 0.3,
  render, // optional: (ctx) => style overrides
}) {
  const spriteCtx = useSprite()
  const { localTime, duration } = spriteCtx
  const exitStart = Math.max(0, duration - exitDur)

  let opacity = 1
  let scale = 1

  if (localTime < entryDur) {
    const t = Easing.easeOutBack(clamp(localTime / entryDur, 0, 1))
    opacity = clamp(localTime / entryDur, 0, 1)
    scale = 0.4 + 0.6 * t
  } else if (localTime > exitStart) {
    const t = Easing.easeInQuad(clamp((localTime - exitStart) / exitDur, 0, 1))
    opacity = 1 - t
    scale = 1 - 0.15 * t
  }

  const overrides = render ? render(spriteCtx) : {}

  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: y,
        width,
        height,
        background: color,
        borderRadius: radius,
        opacity,
        transform: `scale(${scale})`,
        transformOrigin: 'center',
        willChange: 'transform, opacity',
        ...overrides,
      }}
    />
  )
}

// ── Stage ───────────────────────────────────────────────────────────────────

export function Stage({
  width = 1280,
  height = 720,
  duration = 10,
  background = '#f5f5f5',
  loop = true,
  autoplay = true,
  persistKey = 'animstage',
  startAt,
  paused = false,
  children,
}) {
  const [time, setTime] = useState(() => {
    if (startAt !== undefined) return clamp(startAt, 0, duration)
    try {
      const v = parseFloat(localStorage.getItem(persistKey + ':t') || '0')
      return isFinite(v) ? clamp(v, 0, duration) : 0
    } catch {
      return 0
    }
  })
  const [playing, setPlaying] = useState(autoplay)
  const [hoverTime, setHoverTime] = useState(null)
  const [scale, setScale] = useState(1)

  const stageRef = useRef(null)
  const rafRef = useRef(null)
  const lastTsRef = useRef(null)
  const playingRef = useRef(playing)
  playingRef.current = playing
  const pausedByUsRef = useRef(false)

  // Persist playhead
  useEffect(() => {
    try {
      localStorage.setItem(persistKey + ':t', String(time))
    } catch {
      // storage unavailable (private mode, quota): the playhead just isn't persisted
    }
  }, [time, persistKey])

  // Scale the canvas to fit the parent block
  useEffect(() => {
    if (!stageRef.current) return
    const el = stageRef.current
    const measure = () => {
      const s = Math.min(el.clientWidth / width, (el.clientHeight - BAR_H) / height)
      setScale(Math.max(0.05, s))
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [width, height])

  // External pause: stop while `paused`, resume afterwards only if we stopped it
  useEffect(() => {
    if (paused) {
      if (playingRef.current) {
        pausedByUsRef.current = true
        setPlaying(false)
      }
    } else if (pausedByUsRef.current) {
      pausedByUsRef.current = false
      setPlaying(true)
    }
  }, [paused])

  // Fullscreen the whole stage (canvas + bar); the ResizeObserver rescales the canvas.
  const [canFullscreen] = useState(
    () => !!(document.fullscreenEnabled || document.webkitFullscreenEnabled)
  )
  const [fullscreen, setFullscreen] = useState(false)
  useEffect(() => {
    const sync = () =>
      setFullscreen(
        (document.fullscreenElement || document.webkitFullscreenElement) === stageRef.current
      )
    document.addEventListener('fullscreenchange', sync)
    document.addEventListener('webkitfullscreenchange', sync)
    return () => {
      document.removeEventListener('fullscreenchange', sync)
      document.removeEventListener('webkitfullscreenchange', sync)
    }
  }, [])
  const toggleFullscreen = () => {
    if (fullscreen) {
      ;(document.exitFullscreen || document.webkitExitFullscreen).call(document)
    } else {
      const el = stageRef.current
      ;(el.requestFullscreen || el.webkitRequestFullscreen).call(el)
    }
  }

  // Animation loop
  useEffect(() => {
    if (!playing) {
      lastTsRef.current = null
      return
    }
    const step = (ts) => {
      if (lastTsRef.current == null) lastTsRef.current = ts
      const dt = (ts - lastTsRef.current) / 1000
      lastTsRef.current = ts
      setTime((t) => {
        let next = t + dt
        if (next >= duration) {
          if (loop) next = next % duration
          else {
            next = duration
            setPlaying(false)
          }
        }
        return next
      })
      rafRef.current = requestAnimationFrame(step)
    }
    rafRef.current = requestAnimationFrame(step)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      lastTsRef.current = null
    }
  }, [playing, duration, loop])

  const togglePlaying = () => {
    pausedByUsRef.current = false
    setPlaying((p) => !p)
  }

  // Keyboard (while the stage has focus): space = play/pause, ← → = seek
  const onKeyDown = (e) => {
    if (e.code === 'Space') {
      e.preventDefault()
      togglePlaying()
    } else if (e.code === 'ArrowLeft') {
      e.preventDefault()
      setTime((t) => clamp(t - (e.shiftKey ? 1 : 0.1), 0, duration))
    } else if (e.code === 'ArrowRight') {
      e.preventDefault()
      setTime((t) => clamp(t + (e.shiftKey ? 1 : 0.1), 0, duration))
    } else if (e.key === '0' || e.code === 'Home') {
      e.preventDefault()
      setTime(0)
    }
  }

  const displayTime = hoverTime != null ? hoverTime : time

  const ctxValue = useMemo(
    () => ({ time: displayTime, duration, playing, setTime, setPlaying }),
    [displayTime, duration, playing]
  )

  return (
    // Clicking anywhere inside focuses the stage, which scopes the keyboard to it.
    <div
      ref={stageRef}
      tabIndex={0}
      onKeyDown={onKeyDown}
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        background: '#0a0a0a',
        fontFamily: SANS,
        outline: 'none',
      }}
    >
      {/* Canvas area — vertically centered in remaining space */}
      <div
        style={{
          flex: 1,
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          overflow: 'hidden',
          minHeight: 0,
        }}
      >
        <svg
          width={width}
          height={height}
          style={{
            transform: `scale(${scale})`,
            transformOrigin: 'center',
            flexShrink: 0,
            boxShadow: '0 20px 60px rgba(0,0,0,0.4)',
            display: 'block',
          }}
        >
          <foreignObject x="0" y="0" width="100%" height="100%">
            {/* The film is authored against browser defaults, not the host page's text styles. */}
            <div
              xmlns="http://www.w3.org/1999/xhtml"
              className="film-canvas"
              style={{
                width,
                height,
                background,
                position: 'relative',
                overflow: 'hidden',
                color: '#000',
                fontSize: 16,
                fontWeight: 400,
                lineHeight: 'normal',
                letterSpacing: 'normal',
                textAlign: 'left',
              }}
            >
              <TimelineContext.Provider value={ctxValue}>{children}</TimelineContext.Provider>
            </div>
          </foreignObject>
        </svg>
      </div>

      {/* Playback bar — stacked below canvas, never overlapping */}
      <PlaybackBar
        time={displayTime}
        duration={duration}
        playing={playing}
        onPlayPause={togglePlaying}
        onReset={() => setTime(0)}
        onSeek={setTime}
        onHover={setHoverTime}
        fullscreen={fullscreen}
        onFullscreen={canFullscreen ? toggleFullscreen : null}
      />
    </div>
  )
}

// ── Playback bar ────────────────────────────────────────────────────────────
// Play/pause, return-to-begin, scrub track, time display, optional fullscreen toggle.
// Uses fixed-width time fields so layout doesn't thrash.

export function PlaybackBar({
  time,
  duration,
  playing,
  onPlayPause,
  onReset,
  onSeek,
  onHover,
  fullscreen = false,
  onFullscreen = null,
}) {
  const trackRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const timeFromEvent = useCallback(
    (e) => {
      const rect = trackRef.current.getBoundingClientRect()
      const x = clamp((e.clientX - rect.left) / rect.width, 0, 1)
      return x * duration
    },
    [duration]
  )

  const onTrackMove = (e) => {
    if (!trackRef.current) return
    const t = timeFromEvent(e)
    if (dragging) {
      onSeek(t)
    } else {
      onHover(t)
    }
  }

  const onTrackLeave = () => {
    if (!dragging) onHover(null)
  }

  const onTrackDown = (e) => {
    setDragging(true)
    const t = timeFromEvent(e)
    onSeek(t)
    onHover(null)
  }

  useEffect(() => {
    if (!dragging) return
    const onUp = () => setDragging(false)
    const onMove = (e) => {
      if (!trackRef.current) return
      const t = timeFromEvent(e)
      onSeek(t)
    }
    window.addEventListener('mouseup', onUp)
    window.addEventListener('mousemove', onMove)
    return () => {
      window.removeEventListener('mouseup', onUp)
      window.removeEventListener('mousemove', onMove)
    }
  }, [dragging, timeFromEvent, onSeek])

  const pct = duration > 0 ? (time / duration) * 100 : 0
  const fmt = (t) => {
    const total = Math.max(0, t)
    const m = Math.floor(total / 60)
    const s = Math.floor(total % 60)
    const cs = Math.floor((total * 100) % 100)
    return `${m}:${String(s).padStart(2, '0')}.${String(cs).padStart(2, '0')}`
  }

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        padding: '8px 16px',
        background: 'rgba(20,20,20,0.92)',
        borderTop: '1px solid rgba(255,255,255,0.08)',
        width: '100%',
        maxWidth: 680,
        alignSelf: 'center',
        borderRadius: 8,
        color: '#f5f5f5',
        fontFamily: SANS,
        userSelect: 'none',
        flexShrink: 0,
      }}
    >
      <IconButton onClick={onReset} title="Return to start (0)">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path
            d="M3 2v10M12 2L5 7l7 5V2z"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        </svg>
      </IconButton>
      <IconButton onClick={onPlayPause} title="Play/pause (space)">
        {playing ? (
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <rect x="3" y="2" width="3" height="10" fill="currentColor" />
            <rect x="8" y="2" width="3" height="10" fill="currentColor" />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M3 2l9 5-9 5V2z" fill="currentColor" />
          </svg>
        )}
      </IconButton>

      {/* Current time: fixed width so it doesn't thrash */}
      <div
        style={{
          fontFamily: MONO,
          fontSize: 12,
          fontVariantNumeric: 'tabular-nums',
          width: 64,
          textAlign: 'right',
          color: '#f5f5f5',
        }}
      >
        {fmt(time)}
      </div>

      {/* Scrub track */}
      <div
        ref={trackRef}
        onMouseMove={onTrackMove}
        onMouseLeave={onTrackLeave}
        onMouseDown={onTrackDown}
        style={{
          flex: 1,
          height: 22,
          position: 'relative',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
        }}
      >
        <div
          style={{
            position: 'absolute',
            left: 0,
            right: 0,
            height: 4,
            background: 'rgba(255,255,255,0.12)',
            borderRadius: 2,
          }}
        />
        <div
          style={{
            position: 'absolute',
            left: 0,
            width: `${pct}%`,
            height: 4,
            background: '#64d2ff',
            borderRadius: 2,
          }}
        />
        <div
          style={{
            position: 'absolute',
            left: `${pct}%`,
            top: '50%',
            width: 12,
            height: 12,
            marginLeft: -6,
            marginTop: -6,
            background: '#fff',
            borderRadius: 6,
            boxShadow: '0 2px 4px rgba(0,0,0,0.4)',
          }}
        />
      </div>

      {/* Duration: fixed width */}
      <div
        style={{
          fontFamily: MONO,
          fontSize: 12,
          fontVariantNumeric: 'tabular-nums',
          width: 64,
          textAlign: 'left',
          color: 'rgba(246,244,239,0.55)',
        }}
      >
        {fmt(duration)}
      </div>
      {onFullscreen && (
        <IconButton onClick={onFullscreen} title={fullscreen ? 'Exit full screen' : 'Full screen'}>
          {fullscreen ? (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path
                d="M5 1.5V5H1.5M9 1.5V5h3.5M5 12.5V9H1.5M9 12.5V9h3.5"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path
                d="M1.5 5V1.5H5M12.5 5V1.5H9M1.5 9v3.5H5M12.5 9v3.5H9"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            </svg>
          )}
        </IconButton>
      )}
    </div>
  )
}

function IconButton({ children, onClick, title }) {
  const [hover, setHover] = useState(false)
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        width: 28,
        height: 28,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: hover ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.04)',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: 6,
        color: '#f5f5f5',
        cursor: 'pointer',
        padding: 0,
        transition: 'background 120ms',
      }}
    >
      {children}
    </button>
  )
}

// ── VideoSprite ─────────────────────────────────────────────────────────────
// Renders a <video> that loops within [start,end] of its source at `speed`,
// kept in sync with the Stage's playhead.
//
//   <VideoSprite src="clip.mp4" start={2} end={5} speed={1}
//     style={{ width: 640, height: 360 }} />

export function VideoSprite({ src, start = 0, end, speed = 1, style, ...rest }) {
  const t = useTime()
  const ref = useRef(null)
  const span = Math.max(0.001, (end ?? start + 1) - start)
  useEffect(() => {
    const v = ref.current
    if (!v || v.readyState < 1) return
    const target = start + ((t * speed) % span)
    if (Math.abs(v.currentTime - target) > 0.05) v.currentTime = target
  }, [t, start, span, speed])
  return (
    <video
      ref={ref}
      src={src}
      muted
      playsInline
      preload="auto"
      style={{ display: 'block', objectFit: 'cover', ...style }}
      {...rest}
    />
  )
}
