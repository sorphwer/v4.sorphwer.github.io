// Hero of the glance post: the 120 s event film, in the frame and caption row of blog.html.
import { useEffect, useRef, useState } from 'react'
import EventFilm from './scene'

export default function Film() {
  const frameRef = useRef(null)
  const stageRef = useRef(null)
  const [paused, setPaused] = useState(false)
  const [canFullscreen] = useState(
    () => !!(document.fullscreenEnabled || document.webkitFullscreenEnabled)
  )

  // The film runs a React render every frame on the page's main thread. Pause it
  // while scrolled away; Stage resumes it only if this was what paused it.
  useEffect(() => {
    const io = new IntersectionObserver((es) => setPaused(!es.some((e) => e.isIntersecting)), {
      threshold: 0.15,
    })
    io.observe(frameRef.current)
    return () => io.disconnect()
  }, [])

  const openFullscreen = () => {
    const el = stageRef.current
    ;(el.requestFullscreen || el.webkitRequestFullscreen).call(el)
  }

  return (
    <section className="film" aria-label="两分钟短片">
      <div className="frame" ref={frameRef}>
        <div
          ref={stageRef}
          className="film-stage"
          role="group"
          aria-label="短片：每一张关闭的工单，都是下一张的答案（120 秒）"
          style={{ height: 'calc(100cqw * 0.5625 + 44px)' }}
        >
          <EventFilm showCaptions startAt={0} paused={paused} />
        </div>
      </div>
      <div className="cap">
        <span>
          两分钟短片：一张新工单，怎样在图里找到半年前的答案。可拖动进度条；点击画面后，空格暂停，←/→
          逐帧。
        </span>
        {canFullscreen && (
          <button type="button" onClick={openFullscreen}>
            单独打开 ↗
          </button>
        )}
      </div>
    </section>
  )
}
