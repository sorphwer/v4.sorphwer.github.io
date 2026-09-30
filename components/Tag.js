import Link from 'next/link'
import kebabCase from '@/lib/utils/kebabCase'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import Image from 'next/image'

// riinosite3 tag design
const CHIP =
  'mt-1 mr-3 rounded border-2 border-solid border-black text-sm font-medium uppercase text-black transition duration-500 ease-out hover:border-primary-500 hover:text-primary-500 dark:border-gray-300 dark:text-gray-300 dark:hover:border-primary-400 dark:hover:text-primary-400'
const SHADED = 'hover:bg-gray-300 hover:dark:bg-gray-500'

function variant(text) {
  switch (kebabCase(text)) {
    case 'notion':
      return {
        className: `${CHIP} ${SHADED} bg-violet-700 px-2`,
        content: (
          <>
            <Image
              className="brightness-0 filter dark:brightness-200 dark:filter"
              src="/static/images/notion.svg"
              width={14}
              height={14}
              alt="Notion Blog"
            />
            {' ' + text.split(' ').join('-')}
          </>
        ),
      }
    case 'mdx':
      return {
        className: `${CHIP} bg-white p-0`,
        content: <Image src="/static/images/mdx.png" width={34} height={14} alt="mdx" />,
      }
    default:
      return {
        className: `${CHIP} ${SHADED} px-2`,
        content: (
          <>
            <FontAwesomeIcon icon="tags" className="text-black dark:text-gray-300 " />
            {' ' + text.split(' ').join('-')}
          </>
        ),
      }
  }
}

/**
 * Tag chip. Links to the tag's page, or with `onClick` becomes a button that
 * receives the tag text (the home page uses it to filter by the tag).
 * `className` is appended to the chip's own classes.
 */
const Tag = ({ text, onClick, className = '' }) => {
  const chip = variant(text)
  const classes = `${chip.className} ${className}`
  if (onClick) {
    return (
      <button
        type="button"
        onClick={() => onClick(text)}
        aria-label={`Filter by ${text}`}
        className={classes}
      >
        {chip.content}
      </button>
    )
  }
  return (
    <Link href={`/tags/${kebabCase(text)}`}>
      <a className={classes}>{chip.content}</a>
    </Link>
  )
}

export default Tag
