import Link from 'next/link'
import kebabCase from '@/lib/utils/kebabCase'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import Image from 'next/image'
// riinosite3 tag design
const Tag = ({ text }) => {
  if (text === 'Notion') {
    return (
      <Link href={`/tags/${kebabCase(text)}`}>
        <a
          className="mt-1 mr-3 rounded border-2 border-solid border-black px-2 text-sm font-medium uppercase
        text-black transition duration-500
        ease-out hover:border-primary-500
        hover:bg-gray-300 hover:text-primary-500 dark:border-gray-300
        dark:text-gray-300 dark:hover:border-primary-400
        hover:dark:bg-gray-500 dark:hover:text-primary-400"
        >
          <Image
            className="brightness-0 filter  dark:brightness-200 dark:filter"
            src="/static/images/notion.svg"
            width={14}
            height={14}
            alt="Notion Blog"
          />

          {/* <FontAwesomeIcon icon="tags" className='dark:text-gray-300 text-black ' /> */}
          {' ' + text.split(' ').join('-')}
        </a>
      </Link>
    )
  } else if (text === 'mdx') {
    return (
      <Link href={`/tags/${kebabCase(text)}`}>
        <a
          className="mt-1 mr-3 rounded border-2 border-solid border-black px-2 text-sm font-medium uppercase
        text-black transition duration-500
        ease-out hover:border-primary-500
        hover:bg-gray-300 hover:text-primary-500 dark:border-gray-300
        dark:text-gray-300 dark:hover:border-primary-400
        hover:dark:bg-gray-500 dark:hover:text-primary-400"
        >
          <Image
            className="brightness-0 filter  dark:brightness-200 dark:filter"
            src="/static/images/logo.png"
            width={14}
            height={14}
            alt="mdx"
          />

          {/* <FontAwesomeIcon icon="tags" className='dark:text-gray-300 text-black ' /> */}
          {' ' + text.split(' ').join('-')}
        </a>
      </Link>
    )
  } else {
    return (
      <Link href={`/tags/${kebabCase(text)}`}>
        <a
          className="mt-1 mr-3 rounded border-2 border-solid border-black px-2 text-sm font-medium uppercase
        text-black transition duration-500
        ease-out hover:border-primary-500
        hover:bg-gray-300 hover:text-primary-500 dark:border-gray-300
        dark:text-gray-300 dark:hover:border-primary-400
        hover:dark:bg-gray-500 dark:hover:text-primary-400"
        >
          <FontAwesomeIcon icon="tags" className="text-black dark:text-gray-300 " />
          {' ' + text.split(' ').join('-')}
        </a>
      </Link>
    )
  }
}

export default Tag
