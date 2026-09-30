import { useState } from 'react'
import { SORTS } from '@/lib/utils/postFacets'
import { ChevronIcon } from './icons'
import SourceMark from './SourceMark'

/** Years listed before "Show all": three rows of the year grid. */
const YEAR_PREVIEW = 9
/** Unselected tags listed before "Show all" (short enough that the sidebar fits without scrolling). */
const TAG_PREVIEW = 7

function Section({
  id,
  title,
  note,
  defaultOpen = false,
  className = '',
  bodyClassName = '',
  children,
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className={`border-t border-gray-200 dark:border-gray-800 ${className}`}>
      <h3>
        <button
          type="button"
          aria-expanded={open}
          aria-controls={id}
          onClick={() => setOpen(!open)}
          className="flex w-full items-center justify-between gap-3 py-3 text-left text-sm text-gray-700 hover:text-gray-900 dark:text-gray-300 dark:hover:text-gray-100"
        >
          <span>
            {title}
            {note && (
              <span className="ml-2 text-xs text-primary-500 dark:text-primary-400">{note}</span>
            )}
          </span>
          <ChevronIcon
            className={`h-4 w-4 shrink-0 transition-transform ${open ? 'rotate-180' : ''}`}
          />
        </button>
      </h3>
      {open && (
        <div id={id} className={`pb-4 ${bodyClassName}`}>
          {children}
        </div>
      )}
    </div>
  )
}

function Option({ type = 'checkbox', name, checked, onChange, count, children }) {
  return (
    <label className="flex cursor-pointer items-center gap-3 py-1.5 text-sm text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-100">
      <input
        type={type}
        name={name}
        checked={checked}
        onChange={onChange}
        className={`h-3.5 w-3.5 border-gray-300 bg-transparent text-primary-500 focus:ring-primary-500 focus:ring-offset-0 dark:border-gray-600 ${
          type === 'checkbox' ? 'rounded' : ''
        }`}
      />
      <span className="min-w-0 flex-1 truncate">{children}</span>
      {count !== undefined && (
        <span className="text-xs tabular-nums text-gray-400 dark:text-gray-500">{count}</span>
      )}
    </label>
  )
}

const selectedNote = (keys) => (keys.length > 0 ? String(keys.length) : null)

function ShowAllButton({ expanded, total, noun, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="mt-2 self-start text-xs text-primary-500 hover:underline dark:text-primary-400"
    >
      {expanded ? 'Show fewer' : `Show all ${total} ${noun}`}
    </button>
  )
}

/**
 * "Filter and sort" controls: sort order, then year / source / tag facets from
 * lib/utils/postFacets. Facet selections are toggled through `onToggle(field, key)`.
 * Years and tags list a preview until "Show all"; expanded, each list scrolls
 * on its own. On xl the panel lives in a height-capped sticky sidebar where the
 * tag list is the part that shrinks, so the rest of the sidebar stays put.
 */
export default function FilterPanel({ facets, filters, onToggle, onSort }) {
  const [allYears, setAllYears] = useState(false)
  const [allTags, setAllTags] = useState(false)
  // Years stay newest first; selected years are always listed.
  const years = allYears
    ? facets.years
    : facets.years.filter((year, i) => i < YEAR_PREVIEW || filters.years.includes(year.key))
  const selectedTags = facets.tags.filter((tag) => filters.tags.includes(tag.key))
  const otherTags = facets.tags.filter((tag) => !filters.tags.includes(tag.key))
  const tags = [...selectedTags, ...(allTags ? otherTags : otherTags.slice(0, TAG_PREVIEW))]

  return (
    <div className="border-b border-gray-200 dark:border-gray-800 xl:flex xl:min-h-0 xl:flex-col">
      <Section
        id="filter-sort"
        title="Sort by"
        note={SORTS.find((sort) => sort.key === filters.sort).label}
      >
        {SORTS.map((sort) => (
          <Option
            key={sort.key}
            type="radio"
            name="post-sort"
            checked={filters.sort === sort.key}
            onChange={() => onSort(sort.key)}
          >
            {sort.label}
          </Option>
        ))}
      </Section>

      <Section
        id="filter-year"
        title="Year"
        note={selectedNote(filters.years)}
        defaultOpen
        bodyClassName="flex flex-col"
      >
        <div
          className={`grid grid-cols-3 gap-1.5 ${allYears ? 'max-h-36 overflow-y-auto pr-2' : ''}`}
        >
          {years.map((year) => {
            const selected = filters.years.includes(year.key)
            return (
              <button
                key={year.key}
                type="button"
                aria-pressed={selected}
                title={`${year.count} post${year.count === 1 ? '' : 's'}`}
                onClick={() => onToggle('years', year.key)}
                className={`rounded-md border py-1 text-xs tabular-nums transition-colors ${
                  selected
                    ? 'border-primary-500 bg-primary-500 text-white dark:border-primary-500'
                    : 'border-gray-200 text-gray-600 hover:border-gray-400 hover:text-gray-900 dark:border-gray-800 dark:text-gray-400 dark:hover:border-gray-600 dark:hover:text-gray-100'
                }`}
              >
                {year.label}
              </button>
            )
          })}
        </div>
        {facets.years.length > YEAR_PREVIEW && (
          <ShowAllButton
            expanded={allYears}
            total={facets.years.length}
            noun="years"
            onClick={() => setAllYears(!allYears)}
          />
        )}
      </Section>

      <Section id="filter-source" title="Source" note={selectedNote(filters.sources)} defaultOpen>
        {facets.sources.map((source) => (
          <Option
            key={source.key}
            checked={filters.sources.includes(source.key)}
            onChange={() => onToggle('sources', source.key)}
            count={source.count}
          >
            <SourceMark source={source.key} />
          </Option>
        ))}
      </Section>

      <Section
        id="filter-tag"
        title="Tag"
        note={selectedNote(filters.tags)}
        defaultOpen
        className="xl:flex xl:min-h-0 xl:flex-col"
        bodyClassName="xl:flex xl:min-h-0 xl:flex-col"
      >
        <div
          className={`xl:min-h-[8rem] xl:overflow-y-auto ${
            allTags ? 'max-h-72 overflow-y-auto pr-2 xl:max-h-none' : ''
          }`}
        >
          {tags.map((tag) => (
            <Option
              key={tag.key}
              checked={filters.tags.includes(tag.key)}
              onChange={() => onToggle('tags', tag.key)}
              count={tag.count}
            >
              {tag.label}
            </Option>
          ))}
        </div>
        {otherTags.length > TAG_PREVIEW && (
          <ShowAllButton
            expanded={allTags}
            total={facets.tags.length}
            noun="tags"
            onClick={() => setAllTags(!allTags)}
          />
        )}
      </Section>
    </div>
  )
}
