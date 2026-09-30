// Hero of the glance post: the 120 s event film, in the frame of blog.html.
import { useEffect, useRef, useState } from 'react'
import { useT } from '@/components/article/lang'
import EventFilm from './scene'

export default function Film() {
  const t = useT()
  const frameRef = useRef(null)
  const [paused, setPaused] = useState(false)

  // The film runs a React render every frame on the page's main thread. Pause it
  // while scrolled away; Stage resumes it only if this was what paused it.
  useEffect(() => {
    const io = new IntersectionObserver((es) => setPaused(!es.some((e) => e.isIntersecting)), {
      threshold: 0.15,
    })
    io.observe(frameRef.current)
    return () => io.disconnect()
  }, [])

  return (
    <section className="film" aria-label={t('Two-minute film', '两分钟短片')}>
      <div className="frame" ref={frameRef}>
        <div
          className="film-stage"
          role="group"
          aria-label={t(
            'Film: every closed ticket answers the next (120 s)',
            '短片：每一张关闭的工单，都是下一张的答案（120 秒）'
          )}
          style={{ height: 'calc(100cqw * 0.5625 + 44px)' }}
        >
          <EventFilm showCaptions startAt={0} paused={paused} />
        </div>
      </div>
    </section>
  )
}
