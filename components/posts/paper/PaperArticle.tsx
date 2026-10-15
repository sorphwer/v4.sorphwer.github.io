import { T, useLang } from '@/components/article/lang'
import Abstract from './sections/abstract'
import S1Intro from './sections/s1-intro'
import S2Related from './sections/s2-related'
import S3Architecture from './sections/s3-architecture'
import S4Algorithm from './sections/s4-algorithm'
import S5Methodology from './sections/s5-methodology'
import S6Results from './sections/s6-results'
import S6External from './sections/s6-external'
import S7Engineering from './sections/s7-engineering'
import S8Discussion from './sections/s8-discussion'

/**
 * Provenance line above the abstract. Title, byline, language toggle and the
 * header rule belong to the post frame (components/article/PostArticle, which
 * also provides LangProvider because the frontmatter has `titleZh`).
 */
function Provenance() {
  return (
    <div className="animate-paper-rise flex flex-wrap items-center gap-x-4 gap-y-2 font-geist-mono text-xs leading-5 text-ds-gray-900">
      <span className="inline-block rounded-full bg-ds-background-200 px-2 py-0.5 shadow-[inset_0_0_0_1px_var(--ds-gray-alpha-500)]">
        Research
      </span>
      <span>Dify · Zendesk Knowledge-Graph Project</span>
      <span>
        commit <span className="text-ds-gray-1000">dc99579</span>
      </span>
      <span className="w-full font-sans text-sm text-gray-500 dark:text-gray-400">
        <T
          en="Golden Query ablation 2026-07-07 · External baselines 2026-07-08 · 600-ticket rerun 2026-07-15"
          zh="Golden Query 消融 2026-07-07 · 外部基线对比 2026-07-08 · 600 工单复跑 2026-07-15"
        />
      </span>
    </div>
  )
}

/**
 * The paper post body, rendered inside PostArticle's reading column (the
 * site's prose measure). Contents navigation is PostArticle's heading rail,
 * which reads the section headings from the DOM. Body text uses the site's
 * prose font and colours (Inter, gray-700 / gray-300); Geist Mono stays for
 * code.
 */
export default function PaperArticle() {
  const lang = useLang()
  return (
    <article
      lang={lang === 'zh' ? 'zh-CN' : 'en'}
      className="font-sans text-gray-700 dark:text-gray-300"
    >
      <Provenance />
      <Abstract />
      <S1Intro />
      <S2Related />
      <S3Architecture />
      <S4Algorithm />
      <S5Methodology />
      <S6Results />
      <S6External />
      <S7Engineering />
      <S8Discussion />
      <footer className="mt-16 border-t border-ds-gray-alpha-400 pt-6 pb-2 font-geist-mono text-xs text-ds-gray-700">
        riino@dify.ai · 2026-07-09 · commit dc99579 · Dify · Zendesk Knowledge-Graph Project
      </footer>
    </article>
  )
}
