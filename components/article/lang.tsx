/**
 * Post-scoped bilingual machinery. A post whose frontmatter has `titleZh` is
 * wrapped in LangProvider by PostArticle, which also puts LangToggle in the
 * post header; the body renders every string twice via `T` / `useT`
 * (English default, Chinese via toggle). The rest of the app stays
 * monolingual, so this deliberately stays a context instead of an i18n
 * framework.
 */

import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'

export type Lang = 'en' | 'zh'

// Named when only the paper was bilingual; kept so stored preferences survive.
const STORAGE_KEY = 'paper.lang'

const LangContext = createContext<Lang>('en')
const SetLangContext = createContext<(lang: Lang) => void>(() => {})

export function LangProvider({ children }: { children: ReactNode }) {
  // SSR renders English; the stored preference applies after hydration.
  const [lang, setLang] = useState<Lang>('en')

  useEffect(() => {
    const stored = window.localStorage.getItem(STORAGE_KEY)
    if (stored === 'zh' || stored === 'en') setLang(stored)
  }, [])

  const set = (next: Lang) => {
    setLang(next)
    window.localStorage.setItem(STORAGE_KEY, next)
  }

  return (
    <LangContext.Provider value={lang}>
      <SetLangContext.Provider value={set}>{children}</SetLangContext.Provider>
    </LangContext.Provider>
  )
}

export function useLang(): Lang {
  return useContext(LangContext)
}

/** Bilingual fragment: renders the branch matching the active language. */
export function T({ en, zh }: { en: ReactNode; zh: ReactNode }) {
  const lang = useLang()
  return <>{lang === 'en' ? en : zh}</>
}

/** Bilingual string helper for attributes and chart labels. */
export function useT(): (en: string, zh: string) => string {
  const lang = useLang()
  return (en, zh) => (lang === 'en' ? en : zh)
}

export function LangToggle() {
  const lang = useLang()
  const setLang = useContext(SetLangContext)

  const btn = (value: Lang, label: string) => (
    <button
      type="button"
      aria-pressed={lang === value}
      onClick={() => setLang(value)}
      className={`inline-flex h-7 cursor-pointer items-center whitespace-nowrap rounded-md px-2.5 text-xs font-medium leading-none transition-colors duration-150 ${
        lang === value
          ? 'bg-white text-gray-900 shadow-sm ring-1 ring-gray-200 dark:bg-gray-900 dark:text-gray-100 dark:ring-gray-700'
          : 'text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-100'
      }`}
    >
      {label}
    </button>
  )

  return (
    <div
      role="group"
      aria-label="Language"
      className="inline-flex gap-0.5 rounded-lg bg-gray-100 p-1 dark:bg-gray-800"
    >
      {btn('en', 'EN')}
      {btn('zh', '中文')}
    </div>
  )
}
